import json
import time
from datetime import datetime
import threading

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def train(net, optimizer, criterion, train_loader):
    """
    训练一个epoch
    """
    net.train()  # 标记模型进入训练状态
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)  # 前向传播
        loss = criterion(outputs, targets)  # 损失函数
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        train_loss += loss.item()  # 加上本批次样本损失和
        _, predicted = torch.max(outputs.data, 1)  # torch.max()会返回两个张量，第一个是值张量、第二个是索引张量
        total += targets.size(0)  # target.size(0)为本批次训练样本个数
        correct += predicted.eq(targets).sum().item()  # 计算本次batch训练预测正确的数量
    return train_loss / total, 1 - correct / total


def test(net, criterion, test_loader):
    # 评估模型
    net.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return test_loss / total, 1 - correct / total


class Timer:
    def __init__(self):
        self._start_time = None
        self._last_time = None
        self.iter_num = 0

    def start(self):
        self._start_time = time.time()
        self._last_time = self._start_time

    def iter(self, epoch, num_epochs):
        self.iter_num += 1
        cur_time = time.time()
        cost_time = '%.3f' % (cur_time - self._last_time) + 's'  # 本epoch训练所费时间
        total_time = cur_time - self._start_time  # 总训练时间
        remaining_time = total_time / self.iter_num * (num_epochs - epoch)  # 预估的剩余训练时间
        self._last_time = cur_time

        if total_time > 60:
            total_time = '%.3f' % (total_time / 60) + 'min'
        else:
            total_time = '%.3f' % total_time + 's'
        if remaining_time > 60:
            remaining_time = '%.3f' % (remaining_time / 60) + 'min'
        else:
            remaining_time = '%.3f' % remaining_time + 's'
        print(f"本epoch训练所费时间：{cost_time}，总训练时间：{total_time}，预估剩余训练时间：{remaining_time}")


class BufferListener:

    def __init__(self):
        self.stop_training = False
        self.listener_thread = threading.Thread(target=self._input_listener)
        self.listener_thread.start()
        pass

    # 输入监听线程函数
    def _input_listener(self):
        input()
        self.stop_training = True

    def need_stop(self):
        return self.stop_training

    def stop(self):
        self.listener_thread.join()


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_exp(name):
    with open("./exp.json", "r", encoding="utf-8") as f:
        exps = json.load(f)
        for exp in exps['exps']:
            if exp['name'] == name:
                return exp
        return None


def write_exp(exp):
    with open("./exp.json", "r", encoding="utf-8") as f:
        exps = json.load(f)
        exist = False
        for i in exps['exps']:
            if i['name'] == exp['name']:
                exist = True
                i['progress'] = exp['progress']
                i['checkpoint'] = exp['checkpoint']
                i['describe'] = exp['describe']
                break
        if not exist:
            exps['exps'].append(exp)
        exps['last_date'] = now()
    with open("./exp.json", "w", encoding="utf-8") as f:
        json.dump(exps, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    write_exp(exp={
        'name': "exp_adam_beta2",
        "progress": "D1-M1-B1-E2",
        "checkpoint": "file.pth",
        "describe": ""
    })
