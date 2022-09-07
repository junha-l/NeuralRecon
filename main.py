import argparse
import os
import time
from typing import OrderedDict

import torch
from torch.utils.data import DataLoader

from config import cfg, update_config
from datasets import find_dataset_def, transforms
from models import NeuralRecon
from ops.comm import *
from utils import SaveScene


@torch.no_grad()
def test(model, dataloader):
    ## 1. find checkpoint file
    saved_models = [fn for fn in os.listdir(cfg.LOGDIR) if fn.endswith(".ckpt")]
    saved_models = sorted(
        saved_models, key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    ckpt = saved_models[0]
    ckpt_file = os.path.join(cfg.LOGDIR, ckpt)
    print("resuming " + str(ckpt_file))

    ## 2. load checkpoint
    ckpt = torch.load(ckpt_file)
    state_dict = ckpt["model"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[len("module.") :]] = v
    model.load_state_dict(new_state_dict, strict=False)
    epoch_idx = ckpt["epoch"]

    dataloader.dataset.tsdf_cashe = {}
    save_mesh_scene = SaveScene(cfg)

    model = model.eval()
    ## 3. iterate over data loader
    for batch_idx, sample in enumerate(dataloader):
        for n in sample["fragment"]:
            print(n)
        # save mesh if SAVE_SCENE_MESH and is the last fragment
        save_scene = cfg.SAVE_SCENE_MESH and batch_idx == len(dataloader) - 1

        start_time = time.time()
        outputs = model(sample, save_scene)
        print(
            "Epoch {}, Iter {}/{}, time = {:3f}".format(
                epoch_idx,
                batch_idx,
                len(dataloader),
                time.time() - start_time,
            )
        )
        # save mesh
        save_mesh_scene(outputs, sample, epoch_idx)
    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", help="experiment configure file name", required=True, type=str
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    # parse arguments and check
    args = parser.parse_args()
    update_config(cfg, args)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print("number of gpus: {}".format(num_gpus))

    # freeze seed
    torch.manual_seed(cfg.SEED)
    torch.cuda.manual_seed(cfg.SEED)

    # Augmentation
    n_views = cfg.TEST.N_VIEWS
    random_rotation = False
    random_translation = False
    paddingXY = 0
    paddingZ = 0

    transform = []
    transform += [
        transforms.ResizeImage((640, 480)),
        transforms.ToTensor(),
        transforms.RandomTransformSpace(
            cfg.MODEL.N_VOX,
            cfg.MODEL.VOXEL_SIZE,
            random_rotation,
            random_translation,
            paddingXY,
            paddingZ,
            max_epoch=cfg.TRAIN.EPOCHS,
        ),
        transforms.IntrinsicsPoseToProjection(n_views, 4),
    ]
    transforms = transforms.Compose(transform)

    # dataset, dataloader
    MVSDataset = find_dataset_def(cfg.DATASET)
    test_dataset = MVSDataset(
        cfg.TEST.PATH,
        "test",
        transforms,
        cfg.TEST.N_VIEWS,
        len(cfg.MODEL.THRESHOLDS) - 1,
    )
    test_dataloader = DataLoader(
        test_dataset,
        cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TEST.N_WORKERS,
        drop_last=False,
    )

    # model
    model = NeuralRecon(cfg)
    model.cuda()

    test(model, test_dataloader)
