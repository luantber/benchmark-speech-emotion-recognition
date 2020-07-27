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

import torch 
class Ravdess(Dataset):
    def __init__(self,
            root_dir="dataset/ravdess",
            folder_audios="Audio_Speech_Actors_01-24" , 
            csv_file="ravdess_dataset.csv",
            train=True,
            transform=None):

        self.folder_audios=folder_audios
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.train = train
        self.transform = transform
        self.num_labels= 8
        if not os.path.isfile(os.path.join(self.root_dir , self.csv_file )):
            print("Creando CSV")
            self.create_csv()
        
        self.dataset = pd.read_csv(os.path.join(self.root_dir , self.csv_file ) )
        
        train=self.dataset.sample(frac=0.8,random_state=200) #random state is a seed value
        test=self.dataset.drop(train.index)

        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        if self.train :
            self.dataset = train
        else:
            self.dataset = test

        self.transform = transform 
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_path = os.path.join( self.dataset.loc[ idx , 'file'] )
        audio, sr = torchaudio.load(audio_path)
        # print ( audio.size() )
        audio = audio.mean(0,True) #to_mono
        
        emotion = self.dataset.loc[ idx , 'emotion'] - 1 # ( 1 , 2, 3 , ... ) -> (0 , 1 , 2 ...)

        # print ( audio.size() )
        if self.transform:

            audio = self.transform(sr)(audio)
            
        return audio, emotion 

    def create_csv(self): #make private
        files = glob.glob(os.path.join(self.root_dir , self.folder_audios )  + '/**/*.wav', recursive=True)
    
        with open(  os.path.join(self.root_dir,self.csv_file) , mode='w') as new_csv_file:
            new_csv_file = csv.writer(new_csv_file, delimiter=',')
            new_csv_file.writerow( [
                "file" ,
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
                new_csv_file.writerow( [f] + features )

    def benchmark(self, model, model_name, batch_size = 32 , gpu = False , write = True , print_matrix = True, postfix=None):
        
        
        data_test = DataLoader( Ravdess(train=self.train,transform=self.transform) , batch_size=batch_size )
        
        matrix = np.zeros((self.num_labels,self.num_labels))
        with torch.no_grad() :
            for batch in data_test:
                audios , labels = batch
                logits = model(audios)
                _, predicted = torch.max(logits,1) # ( tensor, dim = 1)
                
                matrix += confusion_matrix(labels,predicted,np.arange(0,self.num_labels))

        matrix = matrix.T
        print ( matrix )

        results = {
            "accuracy" : metrics.accuracy(matrix),
            "precision" : metrics.precision(matrix),
            "f1-score" : metrics.f1(matrix,"weight")
        }


        bench = Benchmark("ravdess",model_name, results,self.train )
        if write: 
            if postfix:
                bench.write(postfix)
            else: 
                bench.write()
        return bench