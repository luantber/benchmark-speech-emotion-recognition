import librosa
import pandas as pd
import numpy as np

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from multiprocessing import Pool
from itertools import chain
import pickle as pk
import matplotlib.pyplot as plt
import random

path = "../ser_datasets/ravdess"

# Generates Windows from a 2d array
def get_windows(mfcc, window=55):
    sld = librosa.util.frame(mfcc, frame_length=window, hop_length=10, axis=-1)
    sld = sld.transpose((2, 0, 1))
    return sld


# Get Mfcc from list(iterrows df )
def audioToMfcc(args):
    _, args = args
    y, sr = librosa.load(path + "/audios/" + args.file)
    y, _ = librosa.effects.trim(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    windows = get_windows(mfcc)
    labels = [int(args.emotion)] * len(windows)

    return list(zip(windows, labels))


def split_set(dataset):
    train = list(chain(*dataset))

    train_x, train_y = zip(*train)

    train_x = np.stack(train_x)

    train_y = np.array(train_y)

    store = (train_x, train_y)
    return store


# train_x , train_y ,  test_x , test_y
def get_dataset(cache=True):

    if not cache:
        files = pd.read_csv(path + "/train.csv")
        # Generates a list of lists of windows that contains  [ ( array, label ) ]
        dataset = process_map(
            audioToMfcc, list(files.iterrows()), max_workers=5, chunksize=10
        )

        # store = (train_x, train_y, test_x, test_y)
        pk.dump(dataset, open("dataset.pk", "wb"))

        return dataset

    else:
        dataset = pk.load(open("dataset.pk", "rb"))
        return dataset


# dataset = get_dataset(cache=False)
# train_x , train_y ,  test_x , test_y = dataset

# print( train_x.shape )
# print( train_y.shape )

# print( test_x.shape )
# print( test_y.shape )
