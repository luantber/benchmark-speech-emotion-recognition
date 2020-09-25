import sys
sys.path.append("../..")

from dataset.ravdess import Ravdess
from models.transforms import to_mfcc_compose
from torch.utils.data import DataLoader
from models.cnn1Dv2.cnn1d import CNN1Dv2


path = "lightning_logs/version_25/checkpoints/epoch=148.ckpt"

model_cnn = CNN1Dv2.load_from_checkpoint(path)


r = Ravdess("../../dataset/ravdess",train=False,transform=to_mfcc_compose)
r.benchmark(model_cnn,"cnn1D")