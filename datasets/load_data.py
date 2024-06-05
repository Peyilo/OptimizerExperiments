import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

_classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


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


def get_label(i):
    return _classes[i]


def load_data(dataset='cifar10', batch_size=100):
    if dataset == 'cifar10':
        # 加载CIFAR10训练集和测试集
        train_dataset = dsets.CIFAR10(root="datasets/files/", train=True, download=True, transform=transform_train)
        test_dataset = dsets.CIFAR10(root="datasets/files/", train=False, download=True, transform=transform_test)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, test_loader
    elif dataset == 'cifar100':
        pass
    else:
        raise 'Invalid dataset'
