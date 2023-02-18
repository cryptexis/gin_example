import yaml
import torch
from data_module import MNISTDataModule
from model import GAN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from arch.discriminators.discriminator import Discriminator
from arch.generators.generator import Generator1, Generator2

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config_path', type=str)

args = parser.parse_args()


def get_generator(args, latent_dim):

    if args['generator_arch'] == 'arch1':
        return Generator1(latent_dim=latent_dim, img_shape=dm.dims, base_depth = args['generator_base_depth'])
    elif args['generator_arch'] == 'arch2':
        return Generator2(latent_dim=latent_dim, img_shape=dm.dims, base_depth = args['generator_base_depth'])
    else:
        raise ValueError(f"No generator architecture for argument value {args['generator_arch']}")


with open(args.config_path, "r") as c:
    config = yaml.load(c, Loader=yaml.FullLoader)



dm = MNISTDataModule(config['data']['path_dataset'], config['data']['batch_size'], config['data']['num_workers'])

generator = get_generator(config['models']['generator'], config['latent_dim'])

discriminator = Discriminator(img_shape=dm.dims)


model = GAN(
    generator=generator,
    discriminator=discriminator,
    latent_dim=config['latent_dim'])
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=5,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model, dm)