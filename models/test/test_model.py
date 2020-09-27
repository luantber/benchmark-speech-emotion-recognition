from ser_bench.datasets.test import TestDataset


class TestModel():
    """
    docstring
    """
    pass

    def __call__(self, a):
        return a + 1

    def train(self, data):
        print("entrenando", data)
