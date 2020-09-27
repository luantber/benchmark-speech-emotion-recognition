from models.test.test_model import TestModel
from ser_bench.challenges.v1 import ravdess_bench

model = TestModel()
benchobject = ravdess_bench(model)
