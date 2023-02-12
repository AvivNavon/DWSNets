import logging
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
import wandb
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.ssl.data.inr_dataset import SineINR2CoefDataset
from experiments.ssl.loss import info_nce_loss
from experiments.utils import (
    common_parser,
    count_parameters,
    get_device,
    set_logger,
    set_seed,
    str2bool,
)
from nn.models import DWSModelForClassification, MLPModelForClassification

set_logger()


@torch.no_grad()
def evaluate(model, projection, loader):
    model.eval()
    loss = 0.0
    correct = 0.0
    total = 0.0
    all_features = []
    all_labels = []
    for batch in loader:
        batch = batch.to(device)
        inputs = (
            tuple(
                torch.cat([w, aug_w])
                for w, aug_w in zip(batch.weights, batch.aug_weights)
            ),
            tuple(
                torch.cat([b, aug_b])
                for b, aug_b in zip(batch.biases, batch.aug_biases)
            ),
        )
        features = model(inputs)
        zs = projection(features)
        logits, labels = info_nce_loss(zs, args.temperature)
        loss += F.cross_entropy(logits, labels, reduction="sum")
        total += len(labels)
        real_bs = batch.weights[0].shape[0]
        pred = logits.argmax(1)
        correct += pred.eq(labels).sum()
        all_features.append(features[:real_bs, :].cpu().numpy().tolist())
        all_labels.extend(batch.label.cpu().numpy().tolist())

    model.train()
    avg_loss = loss / total
    avg_acc = correct / total

    return dict(
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        features=np.concatenate(all_features),
        labels=np.array(all_labels),
    )


def main(
    path,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    eval_every: int,
):
    # load dataset
    train_set = SineINR2CoefDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        augmentation=args.augmentation,
        permutation=args.permutation,
        statistics_path=args.statistics_path,
    )
    val_set = SineINR2CoefDataset(
        path=path,
        split="val",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    test_set = SineINR2CoefDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    point = train_set.__getitem__(0)
    weight_shapes = tuple(w.shape[:2] for w in point.weights)
    bias_shapes = tuple(b.shape[:1] for b in point.biases)

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model = {
        "dwsnet": DWSModelForClassification(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=args.dim_hidden,
            n_hidden=args.n_hidden,
            reduction=args.reduction,
            n_classes=args.embedding_dim,
            n_fc_layers=args.n_fc_layers,
            set_layer=args.set_layer,
            n_out_fc=args.n_out_fc,
            dropout_rate=args.do_rate,
            bn=args.add_bn,
        ).to(device),
        "mlp": MLPModelForClassification(
            in_dim=sum([w.numel() for w in weight_shapes + bias_shapes]),
            hidden_dim=args.dim_hidden,
            n_hidden=args.n_hidden,
            n_classes=args.embedding_dim,
            bn=args.add_bn,
        ).to(device),
    }[args.model]

    projection = nn.Sequential(
        nn.Linear(args.embedding_dim, args.embedding_dim),
        nn.ReLU(),
        nn.Linear(args.embedding_dim, args.embedding_dim),
    ).to(device)

    logging.info(f"number of parameters: {count_parameters(model)}")

    optimizer = {
        "adam": torch.optim.Adam(
            [
                dict(
                    params=list(model.parameters()) + list(projection.parameters()),
                    lr=lr,
                ),
            ],
            lr=lr,
            weight_decay=5e-4,
        ),
        "sgd": torch.optim.SGD(
            list(model.parameters()) + list(projection.parameters()),
            lr=lr,
            weight_decay=5e-4,
            momentum=0.9,
        ),
    }[args.optim]

    epoch_iter = trange(epochs)

    criterion = nn.CrossEntropyLoss()
    best_val_loss = 1e6
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch = batch.to(device)
            inputs = (
                tuple(
                    torch.cat([w, aug_w])
                    for w, aug_w in zip(batch.weights, batch.aug_weights)
                ),
                tuple(
                    torch.cat([b, aug_b])
                    for b, aug_b in zip(batch.biases, batch.aug_biases)
                ),
            )
            features = model(inputs)
            zs = projection(features)
            logits, labels = info_nce_loss(zs, args.temperature)
            loss = criterion(logits, labels)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

            if args.wandb:
                log = {
                    "train/loss": loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            )

        if (epoch + 1) % eval_every == 0:
            val_loss_dict = evaluate(model, projection, val_loader)
            test_loss_dict = evaluate(model, projection, test_loader)
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            best_val_criteria = val_loss <= best_val_loss

            if best_val_criteria:
                best_val_loss = val_loss
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict

            if args.wandb:
                log = {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "epoch": epoch,
                }
                if (epoch + 1) % (eval_every * 1) == 0:
                    train_loss_dict = evaluate(model, projection, train_loader)

                    reg = LinearRegression().fit(
                        train_loss_dict["features"], train_loss_dict["labels"]
                    )
                    preds_test = reg.predict(test_loss_dict["features"])
                    preds_val = reg.predict(val_loss_dict["features"])

                    reg_mse_loss = np.square(
                        test_loss_dict["labels"] - preds_test
                    ).mean()
                    reg_mae_loss = np.abs(test_loss_dict["labels"] - preds_test).mean()

                    val_reg_mse_loss = np.square(
                        val_loss_dict["labels"] - preds_val
                    ).mean()
                    val_reg_mae_loss = np.abs(
                        val_loss_dict["labels"] - preds_val
                    ).mean()

                    if args.embedding_dim == 2:
                        low_dim_features = test_loss_dict["features"]
                    else:
                        low_dim_features = TSNE(
                            n_components=2, random_state=42
                        ).fit_transform(test_loss_dict["features"])

                    data = [
                        [*x, *y]
                        for (x, y) in zip(low_dim_features, test_loss_dict["labels"])
                    ]
                    table = wandb.Table(
                        data=data, columns=["f1", "f2", "label1", "label2"]
                    )
                    df = pd.DataFrame(data, columns=["f1", "f2", "label1", "label2"])
                    fig, ax = plt.subplots()
                    extra_params = dict(
                        palette="RdBu"
                    )  # sns.cubehelix_palette(as_cmap=True))
                    sns.scatterplot(
                        data=df,
                        x="f1",
                        y="f2",
                        hue="label1",
                        size="label2",
                        ax=ax,
                        **extra_params,
                    )

                    log.update(
                        {
                            "test/scatter": wandb.Image(plt),
                            "test/reg_mse_loss": reg_mse_loss,
                            "test/reg_mae_loss": reg_mae_loss,
                            "val/reg_mse_loss": val_reg_mse_loss,
                            "val/reg_mae_loss": val_reg_mae_loss,
                            "pred_table": table,
                        }
                    )
                    plt.close(fig)
                wandb.log(log)


if __name__ == "__main__":
    parser = ArgumentParser("SSL trainer", parents=[common_parser])
    parser.set_defaults(
        data_path="dataset/ssl_splits.json",
        lr=5e-3,
        n_epochs=500,
        batch_size=512,
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="sine2coef",
        choices=["sine2coef"],
        help="data name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cannibal",
        choices=["cannibal", "baseline", "re-basin", "inr2vec", "wsl"],
        help="model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="temperature",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="embedding dimension",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="optimizer",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=16,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="num hidden layers",
    )
    parser.add_argument(
        "--n-fc-layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--n-out-fc",
        type=int,
        default=1,
        help="num linear layers at final layer (invariant block)",
    )
    parser.add_argument(
        "--set-layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="number of attention heads",
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default="dataset/statistics.pth",
        help="path to dataset statistics",
    )
    parser.add_argument(
        "--split-path",
        type=str,
        default="dataset/splits.json",
        help="path to dataset statistics",
    )
    parser.add_argument("--eval-every", type=int, default=1, help="eval every")
    parser.add_argument(
        "--augmentation", type=str2bool, default=False, help="use augmentation"
    )
    parser.add_argument(
        "--permutation", type=str2bool, default=False, help="use permutations"
    )
    parser.add_argument(
        "--normalize", type=str2bool, default=False, help="normalize data"
    )

    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add-bn", type=str2bool, default=True, help="add batch norm layers"
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"model_embedding_{args.model}_lr_{args.lr}_hid_dim_{args.dim_hidden}_reduction_{args.reduction}"
            f"_bs_{args.batch_size}_seed_{args.seed}"
        )
        wandb.init(
            project="cannibal-nets",
            entity="ax2",
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    main(
        path=args.data_path,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        device=device,
    )
