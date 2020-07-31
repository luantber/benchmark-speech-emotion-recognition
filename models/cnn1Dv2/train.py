import sys
sys.path.append("../..")

from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler
from models.transforms import to_mfcc_compose
from cnn1d import CNN1Dv2

model = CNN1Dv2()

# profiler = AdvancedProfiler()


# t = Trainer(gpus=1,max_epochs=300,resume_from_checkpoint="lightning_logs/version_19/checkpoints/epoch=80.ckpt")
t = Trainer(gpus=1,max_epochs=150,profiler=True)
t.fit(model)

