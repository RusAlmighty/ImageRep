import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from config import ModelConfig
from data_pipeline import ImageFitting, collate_1d

DIM = 128


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30, residual=False):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.residual = residual
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        sin = torch.sin(self.omega_0 * self.linear(input))
        if self.residual:
            sin += input
        return sin

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(LightningModule):
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        x, y, meta = batch
        out = self(x)
        loss = F.mse_loss(out, y, reduction="none")
        loss = torch.mean(self.pixel_dropout(loss))
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.pixel_dropout = nn.Dropout(p=config.drop_out)
        self.net = []
        self.net.append(SineLayer(config.in_features, config.hidden_features,
                                  is_first=True, omega_0=config.first_omega_0))

        for i in range(config.hidden_layers):
            self.net.append(SineLayer(config.hidden_features, config.hidden_features,
                                      is_first=False, omega_0=config.hidden_omega_0, residual=(i % 2 == 1)))

        if config.outermost_linear:
            final_linear = nn.Linear(config.hidden_features, config.out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / config.hidden_features) / config.hidden_omega_0,
                                             np.sqrt(6 / config.hidden_features) / config.hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(config.hidden_features, config.out_features,
                                      is_first=False, omega_0=config.hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    def validation_step(self, batch, batch_idx, test_flag=False):
        x, y, meta = batch
        out = self(x)
        loss = F.mse_loss(out, y, reduction="mean")
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx, test_flag=False):
        x, y, meta = batch
        out = self(x)
        loss = F.mse_loss(out, y, reduction="mean")
        self.log('test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def train_dataloader(self):
        return DataLoader(ImageFitting(self.config.train_dataset_path, limit=self.config.limit),
                          batch_size=self.config.batch_size,
                          collate_fn=collate_1d, shuffle=True, num_workers=self.config.train_workers)

    def val_dataloader(self):
        return DataLoader(ImageFitting(self.config.validation_dataset_path), batch_size=4,
                          collate_fn=collate_1d)

    def configure_optimizers(self):
        optimaizer = AdamW(self.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimaizer, step_size=self.config.schedule_step)
        return [optimaizer], [lr_scheduler]
