from ser_bench.datasets.test import TestDataset
from ser_bench.utils.benchmark import Benchmark


def ravdess_bench(model):
    dataset = TestDataset(mode="train")
    # results = model()
    return Benchmark()
