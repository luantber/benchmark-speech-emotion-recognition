from models.vgg.cnn_model import CNNModel
from ser_bench.challenges.v1 import ravdess_bench
from models.vgg.prepro import preprocesar_audio


model = CNNModel.load_from_checkpoint(
    'models/vgg/lightning_logs/version_1/checkpoints/epoch=13.ckpt')

resultados = ravdess_bench(model, "VGG", preprocesar_audio)
resultados.write()
