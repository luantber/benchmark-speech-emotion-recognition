from ser_bench.datasets.ravdess import RavdessDataset


data = RavdessDataset(mode="train")

for x, y in data:
    print(x,y)
    break
