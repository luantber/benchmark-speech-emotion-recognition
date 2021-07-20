from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from ser_bench.datasets.ravdess import RavdessDataset
from models.cnn_small.cnn_model import CNNModel
import pytorch_lightning as pl
from models.cnn_small.prepro import preprocesar_audio

train_loader = DataLoader(RavdessDataset(
    mode="train", transform=preprocesar_audio), batch_size=64, num_workers=4)
test_loader = DataLoader(RavdessDataset(
    mode="test", transform=preprocesar_audio), batch_size=64, num_workers=4)


model = CNNModel()


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
    gpus=1, default_root_dir="models/cnn_small",
    # resume_from_checkpoint='models/cnn/lightning_logs/version_0/checkpoints/epoch=23.ckpt',
    max_epochs=50,
    # row_log_interval=10,
    checkpoint_callback=checkpoint_callback

)


trainer.fit(model, train_loader, test_loader)
