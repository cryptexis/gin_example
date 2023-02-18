import gin
import torch
from data_module import MNISTDataModule
from model import GAN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from arch.discriminators.discriminator import Discriminator
from arch.generators.generator import Generator1, Generator2
from gin.torch import external_configurables
from gin_macros import latent_dim

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str)

args = parser.parse_args()

@gin.configurable
def run_pipeline(
    model,
    data_module       
):

    model = model(data_dims=data_module.dims)

    trainer = Trainer(
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=5,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
    )

    trainer.fit(model, data_module)


gin.parse_config_file(args.config_path)

run_pipeline()



