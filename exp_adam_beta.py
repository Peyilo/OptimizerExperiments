"""
探讨Adam优化器的两个beta参数的影响
数据集：cifar10、cifar100
模型：vgg11、vgg16、resnet18、resnet34、googlelenet、densenet
"""
import pandas as pd
import torch
import torch.optim as optim
from torch.backends import cudnn

from datasets.load_data import load_data
from model import *
from utils import *

exp_name = 'exp_adam_beta'
describe = "探讨Adam优化器的两个beta参数的影响"

D = 1
M = 1
B = 1
E = 1

lr = 1e-3
num_epochs = 100
batch_size = 100

timer = Timer()
listener = BufferListener()
models = ['vgg11', 'vgg16', 'resnet18', 'resnet34']
datasets = ['cifar10', 'cifar100']
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
    else:
        raise 'Unknown model'


def calc_betas(n_item):
    beta1 = 1 - 1 / n_item[0]
    beta2 = 1 - 1 / n_item[1]
    return beta1, beta2


n_values = [10, 100, 1000, 10000]
n_items = []  # 由n确定beta的取值，这里生成了16组betas取值
for n1 in n_values:
    for n2 in n_values:
        n_items.append((n1, n2))

# 加载实验进度
exp = load_exp(exp_name)
checkpoint = None
if exp is not None:
    progress = exp['progress'].split('-')
    D = int(progress[0][1:])
    M = int(progress[1][1:])
    B = int(progress[2][1:])
    E = int(progress[3][1:])
    if len(exp['checkpoint']) != 0:
        checkpoint = exp['checkpoint']

# 跳过已经进行过的实验
for i in range(D - 1):
    del datasets[0]
for i in range(M - 1):
    del models[0]
for i in range(B - 1):
    del n_items[0]

for dataset in datasets:
    train_loader, test_loader = load_data(dataset=dataset, batch_size=batch_size)
    for model_name in models:
        for n_item in n_items:
            net = get_model(model_name)
            if device == 'cuda':
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
            start_epoch = 1
            result = {"train_loss": [], "train_error": [], "test_loss": [], "test_error": []}
            betas = calc_betas(n_item)
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)

            # 加载checkpoint的模型参数
            if checkpoint is not None:
                checkpoint = torch.load(f'{checkpoint}.pth')
                net.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                result = checkpoint["eval"]
                start_epoch = E
                checkpoint = None

            # learning_rates = []
            timer.start()
            print(f"> [Start] {model_name}, Adam, betas({betas[0]}, {betas[1]}), n_item({n_item[0]}, {n_item[1]})")
            for epoch in range(start_epoch, num_epochs + 1):
                if listener.need_stop():
                    path = f"{root}temp/{exp_name}/checkpoint"
                    checkpoint = {
                        "model_state_dict": net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval": result
                    }
                    torch.save(checkpoint, f'{path}.pth')
                    write_exp({
                        "name": exp_name,
                        "describe": describe,
                        "progress": f"D{D}-M{M}-B{B}-E{E}",
                        "checkpoint": path
                    })
                    print("终止训练")
                    listener.stop()
                    exit(0)
                print(f'[Epoch] {epoch} / {num_epochs}')
                train_loss, train_error = train(net, optimizer, criterion, train_loader)
                test_loss, test_error = test(net, criterion, test_loader)
                result['train_loss'].append(train_loss)
                result['train_error'].append(train_error)
                result['test_loss'].append(test_loss)
                result['test_error'].append(test_error)
                print("Training Loss: %.5f, Training Error: %.5f." % (train_loss, train_error))
                print("Testing Loss: %.5f, Testing Error: %.5f." % (test_loss, test_error))
                timer.iter(epoch, num_epochs)
                E += 1

            # 保存数据
            df = pd.DataFrame(result)
            df.to_csv(f'{root}temp/{exp_name}/{model_name}_{dataset}_nitem=({n_item[0]}, {n_item[1]}).csv', index=False)
            torch.save(net.state_dict(), f"{root}temp/{exp_name}/{model_name}_{dataset}_nitem=({n_item[0]}, {n_item[1]})_epoch{num_epochs}.pth")
            B += 1
            E = 1
        M += 1
        B = 1
        E = 1
    D += 1
    M = 1
    B = 1
    E = 1
listener.stop()
