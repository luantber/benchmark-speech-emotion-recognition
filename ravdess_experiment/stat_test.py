import numpy as np
from scipy.stats import wilcoxon

files = ["matrix_1.npy", "matrix_2.npy", "matrix_3.npy"]


f1 = np.load("matrix_1.npy")[:, :6, :]
f2 = np.load("matrix_2.npy")[:, :3, :]
f3 = np.load("matrix_3.npy")

# print(f1.shape)
# print(f2.shape)
# print(f3.shape)

fs = np.concatenate([f1, f2, f3], axis=1)
print(fs.shape)
# fs = fs.mean(axis=2)
print(fs.shape)


modelA = fs[0]
modelB = fs[1]
modelC = fs[2]

p_list = []
stat_list = []

model_1 = []
model_2 = []


for i in range(modelA.shape[0]):
    stat, p = wilcoxon(modelB[i], modelC[i])
    # stat_2, p_2 = wilcoxon(modelB[i], modelC[i])

    if p > 0.05:
        p_list.append(p)
        stat_list.append(stat)

        model_1.append(np.array(modelB[i]).mean())
        model_2.append(np.array(modelC[i]).mean())

    # p_list_2.append(p_2)
    # stat_list_2.append(stat_2)
p_list = np.array(p_list)
stat_list = np.array(stat_list)
model_1 = np.array(model_1)
model_2 = np.array(model_2)

print(p_list.mean(), p_list.std())
print(stat_list.mean(), stat_list.std())

# print(modelB.mean(), modelB.std())
# print(modelC.mean(), modelC.std())

print(model_1.mean(), model_1.std())
print(model_2.mean(), model_2.std())

print(len(model_1) / 39.0)

# import matplotlib.pyplot as plt

# # plt.hist(stat_list, bins=12)
# # plt.show()
# plt.xticks(np.arange(0, 10, 0.05), rotation=90)

# plt.xlabel("p_value")
# plt.ylabel("Distribución")
# plt.hist(p_list, bins=20, align="left")
# plt.show()
