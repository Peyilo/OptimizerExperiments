import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data_folder = '../../data/exp_weight_decay_gradient_norm/'
sava_path = '../../result/exp_weight_decay_gradient_norm/'
data_files = {
    "Adam WD=1e-3": "resnet18_cifar10_adam_weightdecay=0.001.csv",
    "Adam WD=5e-4": "resnet18_cifar10_adam_weightdecay=0.0005.csv",
    "Adam WD=1e-4": "resnet18_cifar10_adam_weightdecay=0.0001.csv",
    "Adam WD=0": "resnet18_cifar10_adam_weightdecay=0.csv"
}

colors = {
    "Adam WD=0": "orange",
    "Adam WD=1e-3": "red",
    "Adam WD=1e-4": "green",
    "Adam WD=5e-4": "blue"
}

train_losses, train_errs, test_losses, test_errs = {}, {}, {}, {}
gradient_norm, squared_gradient_norm = {}, {}
for n_item, file in data_files.items():
    reader = pd.read_csv(data_folder + file)
    data = np.array(reader.values.tolist())
    train_losses[n_item] = data[:, 0]
    train_errs[n_item] = data[:, 1]
    test_losses[n_item] = data[:, 2]
    test_errs[n_item] = data[:, 3]
    gradient_norm[n_item] = data[:, 4]
    squared_gradient_norm[n_item] = data[:, 5]

epochs = range(1, 100 + 1)

plt.rcParams['figure.figsize'] = (10, 8.0)
plt.rcParams['image.cmap'] = 'gray'
axes = plt.gca()
axes.set_xlim([0, 100])
for n_item, data in gradient_norm.items():
    plt.plot(epochs, data, label=n_item, color=colors[n_item])
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.title('Gradient Norm')
plt.legend()
plt.show()


plt.subplot()
axes = plt.gca()
axes.set_ylim([0, 5000])
axes.set_xlim([0, 100])
for n_item, data in squared_gradient_norm.items():
    plt.plot(epochs, data, label=n_item,  color=colors[n_item])
plt.grid()
plt.title('Squared gradient Norm')
plt.xlabel("Epochs")
plt.ylabel("Squared gradient Norm")
plt.legend()
plt.show()


