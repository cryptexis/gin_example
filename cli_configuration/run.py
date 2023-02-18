import os
import torch
from data_module import MNISTDataModule
from model import GAN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from arch.discriminators.discriminator import Discriminator
from arch.generators.generator import Generator1, Generator2

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--path_dataset', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_workers', type=int)
parser.add_argument('--latent_dim', type=int)
parser.add_argument('--generator_base_depth', type=int)
parser.add_argument('--generator_arch', type=str)

args = parser.parse_args()


def get_generator(args):

    if args.generator_arch == 'arch1':
        return Generator1(latent_dim=args.latent_dim, img_shape=dm.dims, base_depth = args.generator_base_depth)
    elif args.generator_arch == 'arch2':
        return Generator2(latent_dim=args.latent_dim, img_shape=dm.dims, base_depth = args.generator_base_depth)
    else:
        raise ValueError(f"No generator architecture for argument value {args.generator_arch}")


dm = MNISTDataModule(args.path_dataset, args.batch_size, args.num_workers)

generator = get_generator(args)

discriminator = Discriminator(img_shape=dm.dims)


model = GAN(
    generator=generator,
    discriminator=discriminator,
    latent_dim=args.latent_dim)
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=5,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model, dm)