import argparse

import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from config import ModelConfig, add_model
from model import Siren


def main():
    parser = argparse.ArgumentParser()
    add_model(parser, ModelConfig)
    args = parser.parse_args()
    config = ModelConfig(**vars(args))

    # config = ModelConfig(hidden_features=512, lr=1E-5, hidden_omega_0=60, wd=1E-2)
    wandb.init(project="Image Representation Experiment", name=config.name, config=config.__dict__)
    wandb_logger = WandbLogger(project="Image Representation Experiment", name=config.name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'output/{config.name}',
        filename='ckpt-{epoch:02d}',
    )
    trainer = Trainer(
        gpus=1,
        auto_lr_find=False,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        check_val_every_n_epoch=10,
        auto_scale_batch_size=False,
        log_every_n_steps=1, logger=wandb_logger,
        gradient_clip_val=0.5,
    )
    model = Siren(config)
    trainer.fit(
        model,
    )


if __name__ == "__main__":
    main()
