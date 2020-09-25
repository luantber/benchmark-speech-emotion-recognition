from dataset.ravdess import Ravdess
from models.transforms import to_spectrogram_compose_norandom
from torch.utils.data import DataLoader
from models.cnn import CNN


path = "lightning_logs/version_17/checkpoints/epoch=181.ckpt"

model_cnn = CNN.load_from_checkpoint(path)


r = Ravdess(train=False,transform=to_spectrogram_compose_norandom)
r.benchmark(model_cnn,"cnn_rect_fine_norand")