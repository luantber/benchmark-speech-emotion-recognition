import torchaudio
import torch

def preprocesar_audio(wav_file):
    audio, sr = torchaudio.load(wav_file)
    audio = audio.mean(0, True)
    spec = torchaudio.transforms.MelSpectrogram()(audio)
    return spec[:, :, :128]
