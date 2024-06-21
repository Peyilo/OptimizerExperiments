"""
模型：VGG16（是否带Batch Norm）
数据集：CIFAR-10
优化器：AdamW
"""


import torch
import torchvision
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pandas as pd

from model import *
from swd_optim import *
from utils import Timer

# 超参数
batch_size = 128
epoch_num = 200

device = "cuda" if torch.cuda.is_available() else "cpu"

# 数据集预处理
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='../datasets/files/', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root='../datasets/files/', train=False, download=True, transform=transform_test)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


def define_models(model):
    if model == 'VGG16WithBN':
        return VGG('VGG16', batch_norm=True)
    elif model == 'VGG16WithoutBN':
        return VGG('VGG16', batch_norm=False)
    else:
        raise 'Unspecified model.'


def optimizers(net, opti_name, lr, weight_decay):
    if opti_name == 'SGD':
        return optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)
    elif opti_name == 'Adam':
        return optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay)
    elif opti_name == 'AdamW':
        return optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay / lr)
    elif opti_name == 'AdamS':
        return AdamS(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
    else:
        raise 'Unspecified optimizer.'


def calculate_grad_norm(net):
    """
    计算梯度范数
    """
    grad_norm = 0.
    squared_grad_norm = 0.
    for p in net.parameters():
        data = p.grad.detach().data
        grad_norm += data.norm(2).item() ** 2
        squared_grad_norm += (data ** 2).norm(2).item() ** 2
    grad_norm = grad_norm ** 0.5
    squared_grad_norm = squared_grad_norm ** 0.5
    return grad_norm, squared_grad_norm


def train(net, optimizer):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss / total, 1 - correct / total


def test(net):
    net.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / total, 1 - correct / total


criterion = nn.CrossEntropyLoss(reduction='mean')

models = ['VGG16WithBN', 'VGG16WithoutBN']
optimizer_list = ['AdamW']
wd_list = [1e-3, 5e-4, 1e-4, 0]
lr_dir = {'AdamW': 1e-3}

timer = Timer()
for model_name in models:
    for wd in wd_list:
        net = define_models(model_name)
        if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True
        optimizer = optimizers(net, 'AdamW', lr=lr_dir['AdamW'], weight_decay=wd)

        lambda_lr = lambda epoch: 0.1 ** (epoch // 80)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
        result = {"train_loss": [], "train_err": [], "test_loss": [], "test_err": [],
                  "grad_norm": [], "squared_grad_norm": []}
        timer.start()
        print(f"> [Start] {model_name}, cifar10, AdamW, weight_decay: {wd}")
        for epoch in range(1, epoch_num + 1):
            print(f'[Epoch] {epoch} / {epoch_num}')
            train_loss, train_err = train(net, optimizer)
            grad_norm, squared_grad_norm = calculate_grad_norm(net)
            test_loss, test_err = test(net)

            result['train_loss'].append(train_loss)
            result['train_err'].append(train_err)
            result['test_loss'].append(test_loss)
            result['test_err'].append(test_err)
            result['grad_norm'].append(grad_norm)
            result['squared_grad_norm'].append(squared_grad_norm)

            print("Training Loss: %.5f, Training Err: %.5f." % (train_loss, train_err))
            print("Testing Loss: %.5f, Testing Err: %.5f." % (test_loss, test_err))
            print("grad norm %.5f, squared grad norm %.5f." % (grad_norm, squared_grad_norm))
            scheduler.step()
            timer.iter(epoch, epoch_num)
        df = pd.DataFrame(result)
        result_dir = ''
        file_name = f'{model_name}_AdamW_LR{0.001}_WD{wd}.csv'
        df.to_csv(f'{file_name}', index=False)


