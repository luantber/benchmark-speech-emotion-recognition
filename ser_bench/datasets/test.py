from os import listdir, path
import pandas as pd
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """
    clase Data Test
    """

    def __init__(self, dataset_dir="datasets/test", mode="train", transforms=None):
        files = listdir(dataset_dir)
        if not files:
            raise Exception(
                "No se encontraron archivos en la carpeta:  " + dataset_dir)
        self.mode = mode
        self.data = []
        if mode == "train":
            self.data = pd.read_csv(dataset_dir+"/train.csv")
        elif mode == "test":
            self.data = pd.read_csv(dataset_dir+"/test.csv")
        else:
            raise Exception("arg mode="+mode +
                            " no reconocido, use train o test")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, 'x'] , self.data.loc[idx, 'y']

    def benchmark(self, model):
        print(model)
