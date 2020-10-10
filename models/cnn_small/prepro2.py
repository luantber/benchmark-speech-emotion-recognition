from ser_bench.datasets.ravdess import RavdessDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


import torch
import librosa


def preprocesar_audio(wav_file):
    audio, sr = librosa.load(wav_file, mono=True)
    spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=400,hop_length=200)
    # spec = torchaudio.transforms.AmplitudeToDB()(spec)
    # return spec[:, :, :128]
    # print(spec.shape)
    return spec[:, :125]


train_loader = DataLoader(RavdessDataset(
    mode="train", transform=preprocesar_audio), batch_size=64, num_workers=4)


for x, y in train_loader:
    # print(x[0])
    print(x.size())
    # plt.imshow(x[6].view(128, -1))
    # break

# plt.show()
# Its taking 3.9 segs real average

# librosa 2m20 s
