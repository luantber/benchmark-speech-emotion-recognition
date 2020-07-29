import sys
sys.path.append("../..")

from dataset.ravdess import Ravdess
from models.transforms import to_mfcc_compose
from torch.utils.data import DataLoader
from models.cnn1D.cnn1d import CNN1D


path = "lightning_logs/version_13/checkpoints/epoch=147.ckpt"

model_cnn = CNN1D.load_from_checkpoint(path)


r = Ravdess("../../dataset/ravdess",train=True,transform=to_mfcc_compose)
r.benchmark(model_cnn,"cnn1D")