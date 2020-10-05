from torch.utils.data import DataLoader
from ser_bench.datasets.ravdess import RavdessDataset
from models.cnn.cnn_model import CNNModel
import pytorch_lightning as pl
from prepro import preprocesar_audio

train_loader = DataLoader(RavdessDataset(
    mode="train", transform=preprocesar_audio), batch_size=64, num_workers=4)
test_loader = DataLoader(RavdessDataset(
    mode="test", transform=preprocesar_audio), batch_size=64, num_workers=4)


model = CNNModel()

trainer = pl.Trainer(
    gpus=1, default_root_dir="models/cnn",
    # resume_from_checkpoint='models/cnn/lightning_logs/version_0/checkpoints/epoch=23.ckpt',
    max_epochs=5)


trainer.fit(model, train_loader, test_loader)
