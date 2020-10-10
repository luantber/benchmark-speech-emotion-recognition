import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


class CNNModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 32, 3)
        self.c2 = nn.Conv2d(32, 32, 4)

        self.f1 = nn.Linear(32*20*20, 2048)
        self.f2 = nn.Linear(2048, 8)

        self.pool = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((3, 3))

    def forward(self, x):  # 128x128
        x = F.relu(self.c1(x))  # 126x126 #-2

        x = self.pool(x)  # 63x63 #/2
        # print(x.size())

        x = F.relu(self.c2(x))  # 60x60 #-3
        x = self.pool2(x)  # 20x20 #/3

        # print("helooo", x.size())

        x = x.view(-1, 32*20*20)
        x = F.relu(self.f1(x))
        x = self.f2(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, predicted = torch.max(y_hat, 1)  # max
        acc = accuracy(predicted, y)

        loss = F.cross_entropy(y_hat, y)

        self.log("train_loss", loss)
        self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, predicted = torch.max(y_hat, 1)  # max
        acc = accuracy(predicted, y)

        loss = F.cross_entropy(y_hat, y)

        self.log("val_loss", loss)
        self.log("val_acc", acc, prog_bar=True)

        return loss
