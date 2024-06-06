import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
    "n1=3, n2=3": "resnet18_cifar10_nitem=(3, 3).csv",
    "n1=3, n2=10": "resnet18_cifar10_nitem=(3, 10).csv",
    "n1=3, n2=100": "resnet18_cifar10_nitem=(3, 100).csv",
    "n1=3, n2=1000": "resnet18_cifar10_nitem=(3, 1000).csv",
    "n1=3, n2=10000": "resnet18_cifar10_nitem=(3, 10000).csv",
    "n1=10, n2=3": "resnet18_cifar10_nitem=(10, 3).csv",
    "n1=10, n2=10": "resnet18_cifar10_nitem=(10, 10).csv",
    "n1=10, n2=100": "resnet18_cifar10_nitem=(10, 100).csv",
    "n1=10, n2=1000": "resnet18_cifar10_nitem=(10, 1000).csv",
    "n1=10, n2=10000": "resnet18_cifar10_nitem=(10, 10000).csv",
    "n1=100, n2=3": "resnet18_cifar10_nitem=(100, 3).csv",
    "n1=100, n2=10": "resnet18_cifar10_nitem=(100, 10).csv",
    "n1=100, n2=100": "resnet18_cifar10_nitem=(100, 100).csv",
    "n1=100, n2=1000": "resnet18_cifar10_nitem=(100, 1000).csv",
    "n1=100, n2=10000": "resnet18_cifar10_nitem=(100, 10000).csv",
    "n1=1000, n2=3": "resnet18_cifar10_nitem=(1000, 3).csv",
"""

data_folder = '../../data/exp_adam_beta/resnet18/'
sava_path = '../../result/exp_adam_beta/'
data_files = {
    "n1=3, n2=1000": "resnet18_cifar10_nitem=(3, 1000).csv",
    "n1=10, n2=1000": "resnet18_cifar10_nitem=(10, 1000).csv",
    "n1=100, n2=1000": "resnet18_cifar10_nitem=(100, 1000).csv"
}


train_losses, train_errs, test_losses, test_errs = {}, {}, {}, {}
for n_item, file in data_files.items():
    reader = pd.read_csv(data_folder + file)
    data = np.array(reader.values.tolist())
    train_losses[n_item] = data[:, 0]
    train_errs[n_item] = data[:, 1]
    test_losses[n_item] = data[:, 2]
    test_errs[n_item] = data[:, 3]

epochs = range(1, 100 + 1)

plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
axes = plt.gca()
axes.set_ylim([.12, .4])
axes.set_xlim([0, 100])
for n_item, data in test_errs.items():
    plt.plot(epochs, data, label=n_item)
plt.grid()
plt.legend()
plt.show()



