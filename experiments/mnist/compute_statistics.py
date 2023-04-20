from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.data import Batch, INRDataset
from experiments.utils import common_parser


def compute_stats(data_path: str, save_path: str, batch_size: int = 10000):
    train_set = INRDataset(path=data_path, split="train", normalize=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=8
    )

    batch: Batch = next(iter(train_loader))
    weights_mean = [w.mean(0) for w in batch.weights]
    weights_std = [w.std(0) for w in batch.weights]
    biases_mean = [w.mean(0) for w in batch.biases]
    biases_std = [w.std(0) for w in batch.biases]

    statistics = {
        "weights": {"mean": weights_mean, "std": weights_std},
        "biases": {"mean": biases_mean, "std": biases_std},
    }

    out_path = Path(save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    torch.save(statistics, out_path / "statistics.pth")


if __name__ == "__main__":
    parser = ArgumentParser("MNIST - generate statistics", parents=[common_parser])
    parser.set_defaults(batch_size=10000, save_path="dataset")
    args = parser.parse_args()

    compute_stats(
        data_path=args.data_path, save_path=args.save_path, batch_size=args.batch_size
    )
