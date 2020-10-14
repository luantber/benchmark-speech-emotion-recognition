"""
    Archivo de Entrenamiento
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from models.vgg.model import VGG
from models.vgg.prepro import preprocesar_audio
from ser_bench.datasets.ravdess import RavdessDataset

train_loader = DataLoader(RavdessDataset(
    mode="train", transform=preprocesar_audio), batch_size=32, num_workers=4)
test_loader = DataLoader(RavdessDataset(
    mode="test", transform=preprocesar_audio), batch_size=32, num_workers=4)


model = VGG()


# DEFAULTS used by the Trainer
checkpoint_callback = ModelCheckpoint(
    # filepath=os.getcwd(),
    # save_top_k=1,
    verbose=True,
    monitor='val_acc',
    mode='max',
    # prefix=''
)


trainer = pl.Trainer(
    gpus=1, default_root_dir="models/vgg",
    # resume_from_checkpoint='models/cnn/lightning_logs/version_0/checkpoints/epoch=23.ckpt',
    max_epochs=5,
    row_log_interval=15,
    checkpoint_callback=checkpoint_callback

)


trainer.fit(model, train_loader, test_loader)
