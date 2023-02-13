import argparse
import logging
import random
from typing import List, Tuple, Union

import numpy as np
import torch


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


common_parser = argparse.ArgumentParser(add_help=False, description="common parser")
common_parser.add_argument(
    "--data-path",
    type=str,
    help="path for dataset",
)
common_parser.add_argument(
    "--n-epochs",
    type=int,
    default=100,
    help="num epochs",
)
# Optimization
common_parser.add_argument("--lr", type=float, default=1e-2, help="inner learning rate")
common_parser.add_argument("--batch-size", type=int, default=32, help="batch size")
# General
common_parser.add_argument(
    "--seed", type=int, default=42, help="seed value for 'set_seed'"
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device id")
common_parser.add_argument(
    "--save-path", type=str, default="./output", help="dir path for output file"
)
# Wandb
common_parser.add_argument(
    "--wandb-project", type=str, default="dws-nets", help="wandb project name"
)
common_parser.add_argument(
    "--wandb-entity", type=str, default=None, help="wandb entity name"
)
common_parser.add_argument("--wandb", dest="wandb", action="store_true")
common_parser.add_argument("--no-wandb", dest="wandb", action="store_false")
common_parser.set_defaults(wandb=False)


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def make_coordinates(
    shape: Union[Tuple[int], List[int]],
    bs: int,
    coord_range: Union[Tuple[int], List[int]] = (-1, 1),
) -> torch.Tensor:
    x_coordinates = np.linspace(coord_range[0], coord_range[1], shape[0])
    y_coordinates = np.linspace(coord_range[0], coord_range[1], shape[1])
    x_coordinates, y_coordinates = np.meshgrid(x_coordinates, y_coordinates)
    x_coordinates = x_coordinates.flatten()
    y_coordinates = y_coordinates.flatten()
    coordinates = np.stack([x_coordinates, y_coordinates]).T
    coordinates = np.repeat(coordinates[np.newaxis, ...], bs, axis=0)
    return torch.from_numpy(coordinates).type(torch.float)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
