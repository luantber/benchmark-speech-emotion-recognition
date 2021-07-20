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
        self.c3 = nn.Conv2d(32, 32, 3)
        self.f1 = nn.Linear(32*14*14, 4096)
        self.f2 = nn.Linear(4096, 8)
        self.pool = nn.MaxPool2d((2, 2))
        self.pool2 = nn.MaxPool2d((2, 2))

    def forward(self, x):  # 128x128
        x = F.relu(self.c1(x))  # 126x126 #-2

        x = self.pool(x)  # 63x63 #/2
        # print(x.size())

        x = F.relu(self.c2(x))  # 60x60 #-3
        x = self.pool(x)  # 30x30 #/2

        x = F.relu(self.c3(x))  # 28x28 #-2
        x = self.pool2(x)  # 14x14
        # print(x.size())

        x = x.view(-1, 32*14*14)
        x = F.relu(self.f1(x))
        x = self.f2(x)

        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        _, predicted = torch.max(y_hat, 1)  # max
        acc = accuracy(predicted, y)

        loss = F.cross_entropy(y_hat, y)
        print("Este es el"loss)

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
