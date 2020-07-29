from dataset.ravdess import Ravdess 
from torch.utils.data import DataLoader
from models.transforms import to_spectrogram_compose
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


class CNN(LightningModule):
    def __init__(self):
        super().__init__()

        self.c1 = nn.Conv2d(1,32,3) 
        self.c2 = nn.Conv2d(32,32,4)
        self.c3 = nn.Conv2d(32,32,3) 
        self.f1 = nn.Linear(32*12*19,4096)
        self.f2 = nn.Linear(4096,8)
        self.pool = nn.MaxPool2d( (2, 2) )
        self.pool2 = nn.MaxPool2d((1, 4) )


    def forward(self,x):  #64x320
        x = F.relu(self.c1(x)) #62x318 #-2
        x = self.pool(x) #31x159 #/2

        x = F.relu(self.c2(x)) #28x156 #-3
        x = self.pool(x) #14x78 #/2

        x = F.relu(self.c3(x)) #12x76 #-2
        x = self.pool2(x) #12x19
        # print( x.size())
        x = x.view(-1,32*12*19) #
        x = F.relu(self.f1(x))
        x = self.f2(x)
        return x
        
    def train_dataloader(self):
        data_transform = to_spectrogram_compose
        ravdess_train = Ravdess( train=True , transform = data_transform )
        return DataLoader(ravdess_train, batch_size=64, num_workers=4,pin_memory=True)

    def configure_optimizers(self):
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print ( x.size() , y.size() )
        logits = self.forward(x)
        # print(logits.size())
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)

        # print( loss.item() )
        
        logs = {'loss': loss}

        return {'loss': loss,'log': logs}

