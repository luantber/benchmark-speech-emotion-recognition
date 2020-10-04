from os import listdir, path
import glob
import pandas as pd
from torch.utils.data import Dataset
import random
import csv


random.seed(0)


class RavdessDataset(Dataset):
    """
    Ravdess Dataset
    """

    def __init__(self, dataset_dir="datasets/ravdess", folder_name='Audio_Speech_Actors_01-24', mode="train", transform=None):
        """
        Abre la carpeta dataset_dir y busca si existen los archivos
            mode: train or test
        """
        self.mode = mode
        self.folder_name = folder_name
        self.dataset_dir = dataset_dir
        self.transform = transform

        """
        Revisar posibles errores
        """
        files_found = listdir(self.dataset_dir)
        # print(files_found)
        if not files_found:
            raise Exception(
                "No se encontraron archivos en la carpeta:  " + path.abspath(self.dataset_dir))
        if not self.folder_name in files_found:
            raise Exception(
                "No se encontraro la carpeta " + self.folder_name + "  " + path.abspath(self.dataset_dir))

        """
            Buscar train.csv y test.csv
        """
        if not 'train.csv' in files_found or not 'test.csv' in files_found:
            print("No se encontró train.csv o test.csv, se generán los archivos")
            self._create_csv()

        self.data = []
        if self.mode == "train":
            self.data = pd.read_csv(self.dataset_dir+"/train.csv")
        elif self.mode == "test":
            self.data = pd.read_csv(self.dataset_dir+"/test.csv")
        else:
            raise Exception("arg mode="+self.mode +
                            " no reconocido, use train o test")

    def __getitem__(self, idx):
        audio = self.data.at[idx, "file"]
        emotion = self.data.at[idx, "emotion"]
        if self.transform:
            return self.transform(audio), emotion
        return audio, emotion

    def __len__(self):
        return len(self.data)

    def _create_csv(self):
        """
            Genera los CSV train.csv y test.csv
        """
        files_wav = glob.glob(path.join(
            self.dataset_dir, self.folder_name) + '/**/*.wav', recursive=True)

        random.shuffle(files_wav)
        train_wav = files_wav[:int(0.8 * len(files_wav))]
        test_wav = files_wav[int(0.8 * len(files_wav)):]

        """
        Train CSV
        """
        with open(path.join(self.dataset_dir, 'train.csv'), mode='w') as new_csv_file:
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

            for f in train_wav:
                filename = path.basename(f)
                features = filename[:-4].split('-')
                features = [int(f)-1 for f in features]
                new_csv_file.writerow([f] + features)

        """
        Test CSV
        """
        with open(path.join(self.dataset_dir, 'test.csv'), mode='w') as new_csv_file:
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

            for f in test_wav:
                filename = path.basename(f)
                features = filename[:-4].split('-')
                features = [int(f)-1 for f in features]
                new_csv_file.writerow([f] + features)
