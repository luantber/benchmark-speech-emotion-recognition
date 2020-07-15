import numpy as np 
import matplotlib.pyplot as plt
import cv2
import librosa 

def get_duration(path):
    y, sr = librosa.load(  path )
    y,index = librosa.effects.trim(y)
    return  librosa.get_duration(y=y, sr=sr)


def get_spec( path,  size=64):
    y, sr = librosa.load(  path )
    # print(sr)
    ## Borramos Silencio AL inicio y Final 
    y,index = librosa.effects.trim(y)
    #
    if ( len(y) > sr * 3   ):
        y = y[:sr*3]    
    

    # print ( len ( y)) 
    print ( librosa.get_duration(y=y, sr=sr) )
    #print(index)

    #plt.figure(figsize=(14, 5))
    #librosa.display.waveplot(y, sr=sr)

    X = librosa.stft(y,n_fft=512)
    # print(X)
    Xdb = librosa.amplitude_to_db(abs(X))
    # plt.figure(figsize=(14, 5))
    #librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='mel')
    # plt.show()


    data = np.array(Xdb)
    #print ( data.shape )
    data = cv2.resize(data,(size,size))
    return data
    # plt.figure(figsize=(5,5))
    # plt.imshow(data)
    # plt.show()





