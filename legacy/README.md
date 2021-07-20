# Benchmark speech emotion recognition techniques
> Created by: Luis Bernal Chahuayo (Universidad Nacional de San Agustín - Arequipa, Perú) 

Benchmark created to evaluate Speech Emotion Recognition Models

State of the art introduces a series of Deep Learning Models to recognize emotions on speech, and they compare themselves against others using available datasets. However it is hard to make comparasions due of the different metrics used by each paper. 

This work aims to compare in a fair way those models using the same benchamrks for everyone.


## Roadmap:
- [X] Benchmark Class
- [X] Dataset Class
- [X] Ravdess Benchmark
- [ ] Iemocap Benchmark
- [X] CNN 2D Traditional Spectrogram
- [X] CNN 1D 
- [ ] LSTM
- [ ] RNN
- [ ] Web to present benchmarks

## Current Benchmarks
### RAVDESS 

|Name                |Training|accuracy|precision|f1-score|Timestamp                  |
|--------------------|--------|--------|---------|--------|---------------------------|
|cnn_rect_fine       |True    |0.9783  |0.9765   |0.9783  |2020-07-13 17:33:45.498310 |
|cnn_rect_fine       |True    |0.9809  |0.9806   |0.9809  |2020-07-13 17:34:29.922851 |
|cnn_rect_fine       |False   |0.7153  |0.7081   |0.7132  |2020-07-13 17:35:05.303679 |
|cnn_rect_fine       |False   |0.6701  |0.6672   |0.6645  |2020-07-13 17:35:14.917757 |
|cnn_rect_fine_norand|False   |0.6354  |0.6296   |0.6266  |2020-07-27 16:10:54.448795 |
|cnn_rect_fine_norand|False   |0.6354  |0.6296   |0.6266  |2020-07-27 16:24:29.332543 |

## Usage Examples

Using Ravdess Dataset as a Pytorch Dataset loaded by Pytorch Dataloader
```python
from dataset.ravdess import Ravdess 

# Using Ravdess Dataset as a Pytorch Dataset
ravdess_train = Ravdess( "../../dataset/ravdess", train=True , transform = data_transform )
data_loader DataLoader(ravdess_train, batch_size=256, num_workers=4,pin_memory=True)
```

Benchmarking Pytorch Model 
```python
# Load model from checkpoing with pytorch_lightning
path = "lightning_logs/version_25/checkpoints/epoch=148.ckpt"
model_cnn = CNN1Dv2.load_from_checkpoint(path)

# Benchmark it
r = Ravdess("../../dataset/ravdess",train=False,transform=to_mfcc_compose)
r.benchmark(model_cnn,"cnn1D")
```

## Requirements
- python3
- cuda 10.0
- torch
- sklearn
- torchaudio
- numpy
- pandas
- pytorch_lightning

### Algo
from models.vgg.model import VGG
from models.vgg.prepro import preprocesar_audio
from ser_bench.datasets.ravdess import RavdessDataset