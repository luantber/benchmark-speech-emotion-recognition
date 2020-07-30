import sys
sys.path.append("../..")

from pytorch_lightning import Trainer
from models.transforms import to_mfcc_compose_random
from cnn1d import CNN1D

model = CNN1D()

t = Trainer(gpus=1,max_epochs=300,resume_from_checkpoint="lightning_logs/version_18/checkpoints/epoch=296.ckpt")
# t = Trainer(gpus=1,max_epochs=150)
t.fit(model)

