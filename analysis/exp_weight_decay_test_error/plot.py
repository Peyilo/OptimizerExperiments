import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data_folder = '../../data/exp_weight_decay_test_error/'
sava_path = '../../result/exp_weight_decay_test_error/'
data_files = {
    "Eq1 λ=5e-6": "resnet18_cifar10_direct_regular_weightdecay=5e-06.csv",
    "Eq1 λ=5e-5": "resnet18_cifar10_direct_regular_weightdecay=5e-05.csv",
    "Eq1 λ=5e-4": "resnet18_cifar10_direct_regular_weightdecay=0.0005.csv",
    "Eq2 λ=5e-4": "resnet18_cifar10_decoupled_weightdecay=0.0005.csv"
}

colors = {
    "Eq1 λ=5e-6": "red",
    "Eq1 λ=5e-5": "blue",
    "Eq1 λ=5e-4": "green",
    "Eq2 λ=5e-4": "orange"
}

train_losses, train_errs, test_losses, test_errs = {}, {}, {}, {}
for n_item, file in data_files.items():
    reader = pd.read_csv(data_folder + file)
    data = np.array(reader.values.tolist())
    train_losses[n_item] = data[:, 0]
    train_errs[n_item] = data[:, 1]
    test_losses[n_item] = data[:, 2]
    test_errs[n_item] = data[:, 3]

epochs = range(1, 200 + 1)

plt.rcParams['figure.figsize'] = (7.5, 6.0)
plt.rcParams['image.cmap'] = 'gray'
axes = plt.gca()
axes.set_ylim([0., .7])
axes.set_xlim([0, 200])
for n_item, data in test_errs.items():
    plt.plot(epochs, data, label=n_item, color=colors[n_item])
plt.xlabel("epoch")
plt.ylabel("Test Error")
plt.grid()
plt.legend()
plt.show()



