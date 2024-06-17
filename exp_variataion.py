"""
探讨Adam优化器的两个beta参数的影响
数据集：cifar10、cifar100
模型：vgg11、vgg16、resnet18、resnet34、googlelenet、densenet
"""
import pandas as pd
import os
import optimizer
import torch
import torch.optim as optim
from torch.backends import cudnn

from datasets.load_data import load_data
from model import *
from utils import *

exp_name = 'exp_optimizers'
describe = "探讨权重衰减强度的影响"


lr = 0.01
num_epochs = 200
batch_size = 100
weight_decay=5e-4

timer = Timer()
models = ['resnet18']
datasets = ['cifar10']
optimizers = ['adams', 'adamw']
criterion = nn.CrossEntropyLoss(reduction='sum')


def get_model(name):
    if name == 'vgg11':
        return get_vgg11(10, True)
    elif name == 'vgg13':
        return get_vgg13(10, True)
    elif name == 'vgg16':
        return get_vgg16(10, True)
    elif name == 'vgg19':
        return get_vgg19(10, True)
    elif name == 'resnet18':
        return get_resnet18(10)
    elif name == 'resnet18var':
        return get_resnet18_var(10)
    else:
        raise 'Unknown model'


def get_optimizer(name, params):
    if name == 'sgd':
        return optim.SGD(params, lr=0.001, weight_decay=weight_decay)
    elif name == 'sgds':
        return optimizer.SGDS(params, lr=0.001, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(params, lr=0.1, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(params, lr=0.1, weight_decay=weight_decay)
    elif name == 'adams':
        return optimizer.AdamS(params, lr=0.1, weight_decay=weight_decay)
    else:
        raise 'Unknown'


for dataset in datasets:
    train_loader, test_loader = load_data(dataset=dataset, batch_size=batch_size)
    for model_name in models:
        for name in optimizers:
            net = get_model(model_name)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
            start_epoch = 1
            result = {"train_loss": [], "train_error": [], "test_loss": [], "test_error": [],
                      "squared_gradient_norm": [], "wd": []}
            optimizer = get_optimizer(name, net.parameters())
            lambda_lr = lambda epoch: 0.1 ** (epoch // 80)  # 控制学习率衰减，每过80个epoch，学习率衰减为之前的0.1倍
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)  # 学习率调度器
            timer.start()
            print(f"> [Start] {model_name}, {dataset}, {name}")
            for epoch in range(start_epoch, num_epochs + 1):
                print(f'[Epoch] {epoch} / {num_epochs}')
                train_loss, train_error = train(net, optimizer, criterion, train_loader)
                squared_gradient_norm = 0.0
                for p in net.parameters():
                    if p.grad is not None:
                        squared_gradient_norm += (p.grad.data ** 2).norm(2).item() ** 2
                squared_gradient_norm = squared_gradient_norm ** 0.5
                result['squared_gradient_norm'].append(squared_gradient_norm)
                if name == 'adamw':
                    wd = 1
                else:
                    wd = 1 / optimizer.get_wd()
                result['wd'].append(wd)
                test_loss, test_error = test(net, criterion, test_loader)
                result['train_loss'].append(train_loss)
                result['train_error'].append(train_error)
                result['test_loss'].append(test_loss)
                result['test_error'].append(test_error)
                print("Training Loss: %.5f, Training Error: %.5f." % (train_loss, train_error))
                print("Testing Loss: %.5f, Testing Error: %.5f." % (test_loss, test_error))
                print(f"squared: {squared_gradient_norm}")
                print(f"wd={wd}")
                timer.iter(epoch, num_epochs)
                scheduler.step()

            # 保存数据
            df = pd.DataFrame(result)
            data_dir = f"{root}data/{exp_name}"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            df.to_csv(f'{root}data/{exp_name}/optimizers_{model_name}_{dataset}_{name}.csv', index=False)
