import sys
sys.path.append("../..")

from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from models.transforms import to_mfcc_compose_random
from cnn1d import CNN1Dv2

model = CNN1Dv2()

profiler = AdvancedProfiler()


# t = Trainer(gpus=1,max_epochs=300,resume_from_checkpoint="lightning_logs/version_18/checkpoints/epoch=296.ckpt")
t = Trainer(gpus=1,max_epochs=1,profiler=profiler)
t.fit(model)

