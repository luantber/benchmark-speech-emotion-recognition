from models.test.test_model import TestModel
from ser_bench.datasets.test import TestDataset
from torch.utils.data import DataLoader

data = TestDataset(mode="train")

model = TestModel()
model.train(DataLoader(data))
