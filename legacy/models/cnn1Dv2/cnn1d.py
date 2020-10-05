from dataset.ravdess import Ravdess 
from torch.utils.data import DataLoader
from models.transforms import to_mfcc_compose
from pytorch_lightning.core.lightning import LightningModule
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F


class CNN1Dv2(LightningModule):
    def __init__(self):
        super().__init__()
        #input 1,40,241

        self.c1 = nn.Conv1d(1,64,(5,241)) 
        # self.c2 = nn.Conv1d(64,64,5) 
        # self.pool = nn.MaxPool1d(5,4)
        self.f1 = nn.Linear(64*36,8)
        
        
    def forward(self,x):  #40x241
        x = F.relu(self.c1(x)) #64x36x1 #-4
        
        x = x.view(-1,64,36) 
        
        x = x.view(-1,64*36) #
        x = self.f1(x)
    
        return x
        
    def train_dataloader(self):
        data_transform = to_mfcc_compose
        ravdess_train = Ravdess( "../../dataset/ravdess", train=True , transform = data_transform )
        return DataLoader(ravdess_train, batch_size=256, num_workers=4,pin_memory=True)

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

