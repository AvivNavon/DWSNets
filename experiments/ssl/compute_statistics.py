from argparse import ArgumentParser
from pathlib import Path

import torch

from experiments.ssl.data.inr_dataset import SineINR2CoefDataset
from experiments.utils import common_parser

parser = ArgumentParser("SSL - generate statistics", parents=[common_parser])
parser.set_defaults(
    batch_size=8000,
    save_path="dataset",
)
args = parser.parse_args()

train_set = SineINR2CoefDataset(path=args.data_path, normalize=False)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set, batch_size=args.batch_size, shuffle=True, num_workers=8
)

batch = next(iter(train_loader))
weights_mean = [w.mean(0) for w in batch.weights]
weights_std = [w.std(0) for w in batch.weights]
biases_mean = [w.mean(0) for w in batch.biases]
biases_std = [w.std(0) for w in batch.biases]

statistics = {
    "weights": {"mean": weights_mean, "std": weights_std},
    "biases": {"mean": biases_mean, "std": biases_std},
}

out_path = Path(args.save_path)
out_path.mkdir(exist_ok=True, parents=True)
torch.save(statistics, out_path / "statistics.pth")
