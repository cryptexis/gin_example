import os
import torch
from data_module import MNISTDataModule
from model import GAN
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar



PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)


dm = MNISTDataModule(PATH_DATASETS, BATCH_SIZE, NUM_WORKERS)
model = GAN(*dm.dims)
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
    max_epochs=5,
    callbacks=[TQDMProgressBar(refresh_rate=20)],
)
trainer.fit(model, dm)