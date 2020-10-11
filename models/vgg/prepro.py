"""
    Archivo con funciones de preprocesamiento
"""
# from ser_bench.datasets.ravdess import RavdessDataset
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt

import torchaudio
# import torch


def preprocesar_audio(wav_file):
    """
        Obtiene un archivo .wav y retorna un spectrograma recortado
    """
    audio, _ = torchaudio.load(wav_file)  # pylint: disable=no-member
    audio = audio.mean(0, True)

    spec = torchaudio.transforms.MelSpectrogram(n_fft=600)(audio)
    spec = torchaudio.transforms.AmplitudeToDB()(spec)

    medio = spec.size()[2] // 3

    return spec[:, :, medio:medio+128]


# train_loader = DataLoader(RavdessDataset(
#     mode="train", transform=preprocesar_audio), batch_size=64, num_workers=4)


# for x, y in train_loader:
#     print(x.size())
#     plt.imshow(x[0].view(128, 128))

# plt.show()
# Its taking 3.9 segs real average
