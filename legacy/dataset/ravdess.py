from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import torchaudio
import pandas as pd
import numpy as np
import glob
import os
import csv
from . import metrics
from .benchmark import Benchmark
import time
import torch
from concurrent import futures


class Ravdess(Dataset):
    def __init__(self,
                 root_dir="dataset/ravdess",
                 folder_audios="Audio_Speech_Actors_01-24",
                 csv_file="ravdess_dataset.csv",
                 aug_file="ravdess_dataset_aug",
                 use_aug=True,
                 reset_aug=False,
                 train=True,
                 transform=None):

        self.folder_audios = folder_audios
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.train = train
        self.transform = transform
        self.num_labels = 8

        self.train_string_aug = "train.pk" if train else "test.pk"

        self.use_aug = use_aug
        self.aug_file = aug_file
        self.reset_aug = reset_aug

        if not os.path.isfile(os.path.join(self.root_dir, self.csv_file)):
            print("Creando CSV")
            self.create_csv()

        self.dataset = pd.read_csv(os.path.join(self.root_dir, self.csv_file))

        # random state is a seed value
        train = self.dataset.sample(frac=0.8, random_state=200)
        test = self.dataset.drop(train.index)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        if self.train:
            self.dataset = train
        else:
            self.dataset = test

        self.audios = []
        self.srs = []
        self.emotions = []

        if self.use_aug:
            if self.reset_aug or not os.path.isfile(os.path.join(self.root_dir, self.aug_file + self.train_string_aug)):
                print("Augment File")
                self.create_aug_file(os.path.join(
                    self.root_dir, self.aug_file + self.train_string_aug))

            self.audios, self.srs, self.emotions = torch.load(
                os.path.join(self.root_dir, self.aug_file + self.train_string_aug))

        else:

            for idx in range(len(self.dataset)):
                audio_path = os.path.join(
                    self.root_dir, self.dataset.loc[idx, 'file'])
                audio, sr = torchaudio.load(audio_path)
                # ( 1 , 2, 3 , ... ) -> (0 , 1 , 2 ...)
                emotion = self.dataset.loc[idx, 'emotion'] - 1
                # print ( audio.size() )
                audio = audio.mean(0, True)  # to_mono

                self.audios.append(audio)
                self.srs.append(sr)
                self.emotions.append(emotion)

    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):

        sr = self.srs[idx]
        audio = self.audios[idx]
        emotion = self.emotions[idx]

        if self.transform and not self.use_aug:
            audio = self.transform(sr)(audio)

        return audio, emotion

    def create_csv(self):  # make private
        files = glob.glob(os.path.join(
            self.root_dir, self.folder_audios) + '/**/*.wav', recursive=True)
        print(self.root_dir)
        print(os.path.join(self.root_dir, self.csv_file))
        with open(os.path.join(self.root_dir, self.csv_file), mode='w') as new_csv_file:
            new_csv_file = csv.writer(new_csv_file, delimiter=',')
            new_csv_file.writerow([
                "file",
                "mode",
                "channel",
                "emotion",
                "intensity",
                "statement",
                "repetition",
                "actor"])

            for f in files:
                filename = os.path.basename(f)
                features = filename[:-4].split('-')
                new_csv_file.writerow([f] + features)

    def create_aug_file(self, ruta):
        '''
            Genera una lista de ondas del mismo tamano size_segs
            tensor: 1xN 
            sr: Sampling Rate

        '''
        def window_wave(tensor, sr, size_segs=1, stride_segs=0.25):
            nuevos_audios = []

            size = size_segs * sr
            stride = int(stride_segs * sr)

            i = 0
            while (i+size < tensor.shape[1]):
                nuevos_audios.append(tensor[:, i:i+size])
                i += stride

            nuevos_audios.append(
                tensor[:, tensor.shape[1]-size:tensor.shape[1]])
            return nuevos_audios

        audios = []
        srs = []
        emotions = []
        for idx in range(len(self.dataset)):
            audio_path = os.path.join(
                self.root_dir, self.dataset.loc[idx, 'file'])
            audio, sr = torchaudio.load(audio_path)
            # ( 1 , 2, 3 , ... ) -> (0 , 1 , 2 ...)
            emotion = self.dataset.loc[idx, 'emotion'] - 1
            # print ( audio.size() )
            audio = audio.mean(0, True)  # to_mono

            audios.append(audio)
            srs.append(sr)
            emotions.append(emotion)

        t1 = time.time()

        new_audios = []
        new_srs = []
        new_emotions = []

        for idx in range(len(audios)):
            audio = audios[idx]
            recortados = window_wave(audio, srs[idx])

            if self.transform:
                new_audios += [self.transform(srs[idx])(r) for r in recortados]
            else:
                new_audios += [recortados]

            new_srs += [srs[idx]]*len(recortados)
            new_emotions += [emotions[idx]]*len(recortados)

        assert (len(new_audios) == len(new_emotions) == len(new_srs))
        torch.save((new_audios, new_srs, new_emotions), ruta)
        print("time:", time.time() - t1)

    '''
        Retorna un objeto Benchmark 
    '''

    def benchmark(self, model, model_name, batch_size=32, gpu=False, write=True, print_matrix=True, postfix=None):

        data_test = DataLoader(Ravdess(
            root_dir=self.root_dir, train=self.train, transform=self.transform), batch_size=batch_size)

        matrix = np.zeros((self.num_labels, self.num_labels))
        with torch.no_grad():
            for batch in data_test:
                audios, labels = batch
                logits = model(audios)
                _, predicted = torch.max(logits, 1)  # ( tensor, dim = 1)

                matrix += confusion_matrix(labels, predicted,
                                           np.arange(0, self.num_labels))

        matrix = matrix.T
        print(matrix)

        results = {
            "accuracy": metrics.accuracy(matrix),
            "precision": metrics.precision(matrix),
            "f1-score": metrics.f1(matrix, "weight")
        }

        bench = Benchmark("ravdess", model_name, results, self.train)
        if write:
            if postfix:
                bench.write(postfix)
            else:
                bench.write()
        return bench
