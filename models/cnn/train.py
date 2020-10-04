from torch.utils.data import DataLoader
from ser_bench.datasets.ravdess import RavdessDataset
from models.cnn.cnn_model import CNNModel
import pytorch_lightning as pl
import torchaudio
import torch


def preprocesar_audio(wav_file):
    audio, sr = torchaudio.load(wav_file)

    audio = audio.mean(0, True)
    spec = torchaudio.transforms.MelSpectrogram()(audio)
    return spec[:, :, :128]


train_loader = DataLoader(RavdessDataset(
    mode="train", transform=preprocesar_audio), batch_size=64, num_workers=4)
test_loader = DataLoader(RavdessDataset(
    mode="test", transform=preprocesar_audio), batch_size=64, num_workers=4)


model = CNNModel()

trainer = pl.Trainer(
    gpus=1, default_root_dir="models/cnn", row_log_interval=10,max_epochs=5)
trainer.fit(model, train_loader, test_loader)

# trainer.fit(model, train_loader)
