from pytorch_lightning import Trainer
from models.cnn import CNN


model_cnn = CNN()
# trainer = Trainer(gpus=1,max_epochs=00 )
trainer = Trainer(gpus=1,max_epochs=300 , resume_from_checkpoint="lightning_logs/version_18/checkpoints/epoch=217.ckpt")
trainer.fit(model_cnn)