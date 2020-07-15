from dataset.ravdess import Ravdess
from models.transforms import to_spectrogram_compose
from torch.utils.data import DataLoader
from models.cnn import CNN


path = "lightning_logs/version_17/checkpoints/epoch=181.ckpt"

model_cnn = CNN.load_from_checkpoint(path)


r = Ravdess(train=False,transform=to_spectrogram_compose)
r.benchmark(model_cnn,"cnn_rect_fine")
r.benchmark(model_cnn,"cnn_rect_fine")


# data_test = DataLoader( Ravdess(train=False,transform=to_spectrogram_compose()) , batch_size=32 )
# for x,y in data_test:
#     print (x.size())
# algo = r[0][0]
# print( algo )
# print ( model_cnn(algo.view(1,1,128,128))  ) 

# print ( r[0][1])