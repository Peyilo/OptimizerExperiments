import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


data_folder = '../../data/exp_weight_decay_gradient_norm/'
lr = 0.001
# LR=0.001
data_files = {
    "Adam WD=1e-3": "resnet18_Adam_LR0.001_WD0.001_grad_norm.csv",
    "Adam WD=5e-4": "resnet18_Adam_LR0.001_WD0.0005_grad_norm.csv",
    "Adam WD=1e-4": "resnet18_Adam_LR0.001_WD0.0001_grad_norm.csv",
    "Adam WD=0": "resnet18_Adam_LR0.001_WD0_grad_norm.csv"
}

colors = {
    "Adam WD=0": "orange",
    "Adam WD=1e-3": "red",
    "Adam WD=1e-4": "green",
    "Adam WD=5e-4": "blue"
}

gradient_norm, squared_gradient_norm = {}, {}
for n_item, file in data_files.items():
    reader = pd.read_csv(data_folder + file)
    data = np.array(reader.values.tolist())
    gradient_norm[n_item] = data[:, 0]
    squared_gradient_norm[n_item] = data[:, 1]

epochs = range(1, 200 + 1)

plt.rcParams['figure.figsize'] = (10, 8.0)
plt.rcParams['image.cmap'] = 'gray'
axes = plt.gca()
axes.set_ylim([0, 7])
axes.set_xlim([0, 200])
for n_item, data in gradient_norm.items():
    plt.plot(epochs, data, label=n_item, color=colors[n_item])
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.title(f'ResNet18 With Adam On CIFAR10, lr={lr}')
plt.legend()
plt.show()


plt.subplot()
axes = plt.gca()
axes.set_ylim([0, 1.4])
axes.set_xlim([0, 200])
for n_item, data in squared_gradient_norm.items():
    plt.plot(epochs, data, label=n_item,  color=colors[n_item])
plt.grid()
plt.title(f'ResNet18 With Adam On CIFAR10, lr={lr}')
plt.xlabel("Epochs")
plt.ylabel("Squared gradient Norm")
plt.legend()
plt.show()


lr = 0.1
# LR=0.1
data_files = {
    "Adam WD=1e-3": "resnet18_Adam_LR0.1_WD0.001_grad_norm.csv",
    "Adam WD=5e-4": "resnet18_Adam_LR0.1_WD0.0005_grad_norm.csv",
    "Adam WD=1e-4": "resnet18_Adam_LR0.1_WD0.0001_grad_norm.csv",
    "Adam WD=0": "resnet18_Adam_LR0.1_WD0_grad_norm.csv"
}

gradient_norm, squared_gradient_norm = {}, {}
for n_item, file in data_files.items():
    reader = pd.read_csv(data_folder + file)
    data = np.array(reader.values.tolist())
    gradient_norm[n_item] = data[:, 0]
    squared_gradient_norm[n_item] = data[:, 1]

plt.subplot()
axes = plt.gca()
axes.set_ylim([0, 7])
axes.set_xlim([0, 200])
for n_item, data in gradient_norm.items():
    plt.plot(epochs, data, label=n_item, color=colors[n_item])
plt.grid()
plt.xlabel("Epochs")
plt.ylabel("Gradient Norm")
plt.title(f'ResNet18 With Adam On CIFAR10, lr={lr}')
plt.legend()
plt.show()


plt.subplot()
axes = plt.gca()
axes.set_ylim([0, 1.4])
axes.set_xlim([0, 200])
for n_item, data in squared_gradient_norm.items():
    plt.plot(epochs, data, label=n_item,  color=colors[n_item])
plt.grid()
plt.title(f'ResNet18 With Adam On CIFAR10, lr={lr}')
plt.xlabel("Epochs")
plt.ylabel("Squared gradient Norm")
plt.legend()
plt.show()
