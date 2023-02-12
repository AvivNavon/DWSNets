import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import wandb
from torch import nn
from tqdm import trange

from experiments.ssl.data.sine_data import SineDataset
from experiments.utils import common_parser, get_device, set_logger, set_seed
from nn.inr import INR

# import matplotlib.pyplot as plt


set_logger()


def main(lr, bs, device):
    # ----
    # Nets
    # ---
    model: nn.Module = INR(
        in_dim=1,
        n_layers=3,
        up_scale=32,
        out_channels=1,
        pe_features=args.pe_features,
    )
    model = model.to(device)

    # ---------
    # Task loss
    # ---------
    criteria = nn.MSELoss()

    # dataset and dataloaders
    coef = np.random.uniform(0.0, 10.0, 2)
    logging.info(f"coefs [{coef.tolist()[0]:.3f}, {coef.tolist()[1]:.3f}]")
    train_dataset = SineDataset(coef=coef, n_samples=args.n_samples)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs)

    # optimizer
    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    for epoch in epoch_iter:
        for batch in train_loader:
            model.train()
            optimizer.zero_grad()
            batch = batch.to(device)

            out = model(batch.x)
            loss = criteria(out, batch.y)
            loss.backward()
            optimizer.step()

        if loss.item() <= args.early_stop_loss:
            break

        if (epoch + 1) % args.eval_every == 0:
            logging.info(f"train loss: {loss.item():.3f}")
            if args.wandb:
                wandb.log({"test/loss": loss.item()})
    logging.info(f"final loss: {loss.item():.5f}")
    out_path = Path(args.save_path)
    out_path.mkdir(exist_ok=True, parents=True)
    sd = model.state_dict()
    sd["coef"] = coef.tolist()
    sd["x"] = train_dataset.x
    sd["y"] = train_dataset.y
    torch.save(sd, out_path / f"model_{args.seed}.pth")

    # plt.scatter(batch.x, batch.y)
    # plt.scatter(batch.x, model(batch.x).flatten().detach().cpu().numpy())
    # plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("Sine INR model trainer", parents=[common_parser])
    parser.set_defaults(
        lr=1e-3,
        n_epochs=1500,
        batch_size=2000,
        save_path="artifacts",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer type",
    )
    parser.add_argument("--early-stop-loss", default=0.0001, type=float)
    parser.add_argument("--pe-features", default=None, type=int)
    parser.add_argument("--eval-every", default=500, type=int)
    parser.add_argument("--n-samples", default=1000, type=int)

    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = f"sine_toy_lr_{args.lr}_bs_{args.batch_size}_seed_{args.seed}"
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=name)

    device = get_device(gpus=args.gpu)
    main(lr=args.lr, bs=args.batch_size, device=device)
