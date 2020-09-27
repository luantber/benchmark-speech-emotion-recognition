from os import listdir, path
from ser_bench.utils import benchmark


class Iemocap(object):
    """
    docstring
    """

    def __init__(self, dataset_dir="datasets/iemocap"):
        try:
            print(listdir(dataset_dir))
        except:
            print("Folder Not Found: ", path.abspath(dataset_dir))

    def benchmark(self, model):
        print(model)
