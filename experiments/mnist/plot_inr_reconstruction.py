from argparse import ArgumentParser

from torchvision.utils import save_image, make_grid
import torch

from experiments.data import INRImageDataset
from experiments.utils import common_parser


parser = ArgumentParser(
        "MNIST - Reconstruct images grid from INRs", parents=[common_parser]
    )
args = parser.parse_args()

dataset = INRImageDataset(
    path=args.data_path,  # path to splits json file
    augmentation=False,
    split="train",
)

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
batch = next(iter(loader))
save_image(batch.image.squeeze(-1), "mnist_grid.png")
