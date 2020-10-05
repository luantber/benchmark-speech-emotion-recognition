from torchvision.transforms import Compose, Normalize
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, MFCC
import torch 


torch.manual_seed(200)

def random_crop(s_tensor):
    # print ( s_tensor.size())
    assert( s_tensor.size(2) >= s_tensor.size(1)*5)
    ran = torch.randint(s_tensor.size(2)-s_tensor.size(1)*5,(1,)).item()
    # print(ran)
    # print(s_tensor[:,:,ran:ran+s_tensor.size(1)].size())
    result = s_tensor[:,:,ran:ran+s_tensor.size(1)*5]
    # print ( result.size() )
    return result 

def rectangular_crop(s_tensor):
    assert( s_tensor.size(2) >= s_tensor.size(1)*5)
    ran = 100
    result = s_tensor[:,:,ran:ran+s_tensor.size(1)*5]
    # print ( result.size() )
    return result 

def rectangular_cropv2(tensor,sr=48000):
    return tensor[ sr:2*sr ]

#sr : Sampling Rate
def to_spectrogram_compose(sr):
    return Compose([
        MelSpectrogram(sr,n_mels=64),
        AmplitudeToDB(),
        random_crop,
        Normalize([-56],[20]) #mean, std
    ])


def to_spectrogram_compose_norandom(sr):
    return Compose([
        MelSpectrogram(sr,n_mels=64),
        AmplitudeToDB(),
        rectangular_crop,
        Normalize([-56],[20])
    ])


def to_mfcc_crop_compose_random(sr):
    return Compose([
        random_crop,
        MFCC( sr  ),
        Normalize([-19],[103])
    ])


def to_mfcc_crop_compose(sr):
    return Compose([
        rectangular_cropv2,
        MFCC( sr  ),
        Normalize([-19],[103])
    ])

def to_mfcc_compose(sr):
    return Compose([
        MFCC( sr  ),
        Normalize([-19],[103])
    ])
