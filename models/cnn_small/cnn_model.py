import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning.metrics.functional import accuracy


class CNNModel(LightningModule):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1, 64, 3)
        self.c2 = nn.Conv2d(64, 64, 3)

        self.c3 = nn.Conv2d(64, 128, 3)
        self.c4 = nn.Conv2d(128, 128, 3)

        self.c5 = nn.Conv2d(128, 256, 3)
        self.c6 = nn.Conv2d(256, 256, 3)

        self.c7 = nn.Conv2d(256, 512, 3)
        self.c8 = nn.Conv2d(512, 512, 3)

        self.c9 = nn.Conv2d(512, 512, 3)
        self.c10 = nn.Conv2d(512, 512, 3)

        self.f1 = nn.Linear(32*20*20, 4096)
        self.f2 = nn.Linear(4096, 4096)
        self.f3 = nn.Linear(4096, 8)

        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x):  # 128x128
        x = F.relu(self.c1(x))  # 126x126 #-2
        x = F.relu(self.c2(x))  # 124x124 #-2
        x = self.pool(x)  # 62x62 #/2

        x = F.relu(self.c3(x))  # 60x60 #-2
        x = F.relu(self.c4(x))  # 58x58 #-2
        x = self.pool(x)  # 29x29 #/2

        x = F.relu(self.c5(x))  # 27x27 #-2
        x = F.relu(self.c6(x))  # 25x25 #-2
        x = self.pool(x)  # 12x12 #/2

        x = F.relu(self.c7(x))  # 10x10 #-2
        x = F.relu(self.c8(x))  # 8x8 #-2
        x = self.pool(x)  # 4x4 #/2

        x = F.relu(self.c9(x))  # 6x6 #-2
        x = F.relu(self.c10(x))  # 4x4 #-2
        x = self.pool(x)  # 2x2 #/2

        x = x.view(-1, 512*2*2)
        x = nn.Dropout()(x)
        x = F.relu(self.f1(x))
        x = nn.Dropout()(x)
        x = F.relu(self.f2(x))
        x = nn.Dropout()(x) 
        x = F.relu(self.f3(x))

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
