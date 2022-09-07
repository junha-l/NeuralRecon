import logging

import torch
import torch.nn as nn
from torchsparse.tensor import PointTensor

from models.modules import SPVCNN
from ops.back_project import back_project
from ops.generate_grids import generate_grid

from .gru_fusion import GRUFusion


class NeuConNet(nn.Module):
    """
    Coarse-to-fine network.
    """

    def __init__(self, cfg):
        super(NeuConNet, self).__init__()
        self.cfg = cfg
        self.n_scales = len(cfg.THRESHOLDS) - 1

        ch_in = [
            80 + 1,          ## MNasNet Level 0 feat (80) + z_axis coord (1)
            96 + 40 + 2 + 1, ## prev feat (96) + MNasNet Level 1 feat (40) + prev tsdf/occ (2) + z_axis coord (1)  
            48 + 24 + 2 + 1, ## prev feat (40) + MNasNet Level 2 feat (24) + prev tsdf/occ (2) + z_axis coord (1)  
        ]
        channels = [96, 48, 24]

        # GRU Fusion
        self.gru_fusion = GRUFusion(cfg, channels)
        # sparse conv & implicit functions
        self.sp_convs = nn.ModuleList()
        self.tsdf_preds = nn.ModuleList()
        self.occ_preds = nn.ModuleList()

        for i in range(len(cfg.THRESHOLDS)):
            self.sp_convs.append(
                SPVCNN(
                    num_classes=1,
                    in_channels=ch_in[i],
                    pres=1,
                    cr=1 / 2**i,
                    vres=self.cfg.VOXEL_SIZE * 2 ** (self.n_scales - i),
                    dropout=self.cfg.SPARSEREG.DROPOUT,
                )
            )
            self.tsdf_preds.append(nn.Linear(channels[i], 1))
            self.occ_preds.append(nn.Linear(channels[i], 1))

    @torch.no_grad()
    def upsample(self, prev_feat, prev_coords, interval, num=8):
        pos_list = [1, 2, 3, [1, 2], [1, 3], [2, 3], [1, 2, 3]]
        n, c = prev_feat.shape
        up_feat = prev_feat.unsqueeze(1).expand(-1, num, -1).contiguous()
        up_coords = prev_coords.unsqueeze(1).repeat(1, num, 1).contiguous()
        for i in range(num - 1):
            up_coords[:, i + 1, pos_list[i]] += interval

        up_feat = up_feat.view(-1, c)
        up_coords = up_coords.view(-1, 4)

        return up_feat, up_coords

    def forward(self, features, inputs, outputs):
        ################################
        # TODO: Implement forward pass #
        ################################

        bs = features[0][0].shape[0]
        prev_feat, prev_coords = None, None

        # ----coarse to fine----
        for i in range(self.cfg.N_LAYER):
            interval = 2 ** (self.n_scales - i)
            scale = self.n_scales - i

            if i == 0:
                # ----generate new coords----
                coords = generate_grid(self.cfg.N_VOX, interval)[0]
                up_coords = []
                for b in range(bs):
                    up_coords.append(
                        torch.cat(
                            [
                                torch.ones(1, coords.shape[-1]).to(coords.device) * b,
                                coords,
                            ]
                        )
                    )
                up_coords = torch.cat(up_coords, dim=1).permute(1, 0).contiguous()
            else:
                # ----upsample coords----
                up_feat, up_coords = self.upsample(prev_feat, prev_coords, interval)

            # ----a. back project----
            feats = torch.stack([feat[scale] for feat in features])
            KRcam = (
                inputs["proj_matrices"][:, :, scale].permute(1, 0, 2, 3).contiguous()
            )
            volume, count = back_project(
                up_coords,
                inputs["vol_origin_partial"],
                self.cfg.VOXEL_SIZE,
                feats,
                KRcam,
            )

            # ----concat feature from last stage----
            if i != 0:
                feat = torch.cat([volume, up_feat], dim=1)
            else:
                feat = volume

            # ----convert to aligned camera coordinate----
            r_coords = up_coords.detach().clone().float()
            for b in range(bs):
                batch_ind = torch.nonzero(up_coords[:, 0] == b).squeeze(1)
                coords_batch = up_coords[batch_ind][:, 1:].float()
                coords_batch = (
                    coords_batch * self.cfg.VOXEL_SIZE
                    + inputs["vol_origin_partial"][b].float()
                )
                coords_batch = torch.cat(
                    (coords_batch, torch.ones_like(coords_batch[:, :1])), dim=1
                )
                coords_batch = (
                    coords_batch
                    @ inputs["world_to_aligned_camera"][b, :3, :]
                    .permute(1, 0)
                    .contiguous()
                )
                r_coords[batch_ind, 1:] = coords_batch
            # batch index is in the last position
            r_coords = r_coords[:, [1, 2, 3, 0]]

            # ----b. sparse conv 3d backbone----
            point_feat = PointTensor(feat, r_coords)
            feat = self.sp_convs[i](point_feat)

            # ----c. gru fusion----
            up_coords, feat, tsdf_target, occ_target = self.gru_fusion(
                up_coords, feat, inputs, i
            )
            grid_mask = torch.ones_like(feat[:, 0]).bool()
            tsdf = self.tsdf_preds[i](feat)
            occ = self.occ_preds[i](feat)

            # ------d. define the sparsity for the next stage-----
            occupancy = occ.squeeze(1) > self.cfg.THRESHOLDS[i]
            occupancy[grid_mask == False] = False

            # ------define feature and coordinate for the next stage-----
            prev_coords = up_coords[occupancy]
            for b in range(bs):
                batch_ind = torch.nonzero(prev_coords[:, 0] == b).squeeze(1)
                if len(batch_ind) == 0:
                    logging.warning("no valid points: scale {}, batch {}".format(i, b))
                    return outputs

            prev_feat = feat[occupancy]
            prev_tsdf = tsdf[occupancy]
            prev_occ = occ[occupancy]

            prev_feat = torch.cat([prev_feat, prev_tsdf, prev_occ], dim=1)

            if i == self.cfg.N_LAYER - 1:
                outputs["coords"] = prev_coords
                outputs["tsdf"] = prev_tsdf

        return outputs
