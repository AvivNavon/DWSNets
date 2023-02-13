import json
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

from experiments.utils import common_parser, set_logger

if __name__ == "__main__":
    parser = ArgumentParser("SSL - generate data splits", parents=[common_parser])
    parser.add_argument(
        "--name", type=str, default="ssl_splits.json", help="json file name"
    )
    parser.add_argument(
        "--val-size", type=int, default=1000, help="number of validation examples"
    )
    parser.set_defaults(
        batch_size=10000,
        save_path="dataset",
    )
    args = parser.parse_args()

    set_logger()
    save_path = Path(args.save_path) / args.name
    inr_path = Path(args.data_path)

    data_split = defaultdict(lambda: list)
    files = [f.as_posix() for f in inr_path.glob("**/*.pth")]
    train, test = train_test_split(files, test_size=args.test_size)
    train, val = train_test_split(train, test_size=args.test_size)

    data_split["train"] = train
    data_split["test"] = test
    data_split["val"] = val

    with open(save_path, "w") as file:
        json.dump(data_split, file)
