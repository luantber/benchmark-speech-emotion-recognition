from models.cnn.cnn_model import CNNModel
from ser_bench.challenges.v1 import ravdess_bench
from models.cnn.prepro import preprocesar_audio


model = CNNModel.load_from_checkpoint(
    'models/cnn/lightning_logs/version_8/checkpoints/epoch=36.ckpt')

resultados = ravdess_bench(model, "CNN" , preprocesar_audio)
resultados.write()
