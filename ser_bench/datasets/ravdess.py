from os import listdir, path
from ser_bench.utils import Benchmark


class Ravdess():
    """
    docstring
    """

    def __init__(self, dataset_dir="datasets/ravdess"):
        try:
            print(listdir(dataset_dir))
        except:
            print("Folder Not Found: ", path.abspath(dataset_dir))

    def benchmark(self, model):
        print(model)
