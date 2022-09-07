import torch
import torch.nn as nn

from utils import tocuda

from .backbone import MnasMulti
from .gru_fusion import GRUFusion
from .neucon_network import NeuConNet


class NeuralRecon(nn.Module):
    """
    NeuralRecon main class.
    """

    def __init__(self, cfg):
        super(NeuralRecon, self).__init__()
        self.cfg = cfg.MODEL
        alpha = float(self.cfg.BACKBONE2D.ARC.split("-")[-1])
        # other hparams
        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)
        self.n_scales = len(self.cfg.THRESHOLDS) - 1

        # networks
        self.backbone2d = MnasMulti(alpha)
        self.neucon_net = NeuConNet(cfg.MODEL)
        # for fusing to global volume
        self.fuse_to_global = GRUFusion(cfg.MODEL, direct_substitute=True)

    def normalizer(self, x):
        """Normalizes the RGB images to the input range"""
        return (x - self.pixel_mean.type_as(x)) / self.pixel_std.type_as(x)

    def forward(self, inputs, save_mesh=False):
        ################################
        # TODO: Implement forward pass #
        ################################

        inputs = tocuda(inputs)
        outputs = {}
        imgs = torch.unbind(inputs["imgs"], 1)

        # image feature extraction
        features = [self.backbone2d(self.normalizer(img)) for img in imgs]

        # coarse-to-fine decoder: SparseConv and GRU Fusion.
        outputs = self.neucon_net(features, inputs, outputs)

        # fuse to global volume.
        if not self.training and "coords" in outputs.keys():
            outputs = self.fuse_to_global(
                outputs["coords"],
                outputs["tsdf"],
                inputs,
                self.n_scales,
                outputs,
                save_mesh,
            )

        return outputs
