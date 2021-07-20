from ser_bench.datasets.ravdess import RavdessDataset
from ser_bench.utils.benchmark import Benchmark
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import f1_score


def ravdess_bench(model, model_name, transform=None):
    dataset = DataLoader(RavdessDataset(
        mode="test", transform=transform), batch_size=64, num_workers=4)

    temp_res = None
    temp_y = None
    for x, y in dataset:
        res = model(x)
        _, res = torch.max(res, 1)  # max
        temp_res = torch.cat((res, temp_res)) if temp_res != None else res
        temp_y = torch.cat((y, temp_y)) if temp_y != None else y

    f1 = f1_score(temp_y, temp_res, average="micro")

    return Benchmark("ravdess", model_name, {"f1": f1}, False)
