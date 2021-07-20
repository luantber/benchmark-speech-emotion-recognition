from models.vgg.model import VGG
from models.vgg.prepro import preprocesar_audio
from ser_bench.challenges.v1 import ravdess_bench


model = VGG.load_from_checkpoint(
    'models/vgg/lightning_logs/version_0/checkpoints/epoch=23.ckpt')

resultados = ravdess_bench(model, "VGG", preprocesar_audio)
resultados.write()
