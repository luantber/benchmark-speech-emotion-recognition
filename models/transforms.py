from torchvision.transforms import Compose, Normalize
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
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

def to_spectrogram_compose(sr):
    return Compose([
        MelSpectrogram(sr,n_mels=64),
        AmplitudeToDB(),
        random_crop,
        Normalize([-56],[20])
    ])