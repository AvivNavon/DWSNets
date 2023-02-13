# Self-supervised learning for dense representation

The INR data is available [here](TODO). Please download the data and place it in, e.g. `dataset/ssl-inrs` folder.

Next, create the data split using:

```shell
python generate_data_splits.py --data-path dataset/ssl-inrs --save-path dataset
```
This will create a json file `dataset/ssl_splits.json`.


The dataset statistics file is available [here](TODO). Please place it in `dataset/statistics.pth`. 
You can also compute the statistics using the `compute_statistics.py` like so:
```shell
python compute_statistics.py --data-path dataset/ssl_splits.json
```

Next, to run the experiment:

```shell
python trainer.py --data-path dataset/ssl_splits.json --statistics-path dataset/statistics.pth --model dwsnet
```

To enable [wandb](https://wandb.ai/site) experiment tracking:

```shell
python trainer.py --data-path dataset/ssl_splits.json --statistics-path dataset/statistics.pth --model dwsnet --wandb --wandb-project dws-nets --wandb-entity <your-entity>
```