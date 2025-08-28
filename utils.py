import collections
from tqdm import tqdm
import numpy as np
import torch
import random
import torch.nn as nn
from typing import Tuple
from collections import Counter
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:  # 检查偏置是否存在
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0, std=0.1)
    elif isinstance(m, nn.RNN) or isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

def quantize(x,input_compress_settings={}):
    compress_settings={'n':2}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n=compress_settings['n']
    #print('n:{}'.format(n))
    x=x.float()
    x_norm=torch.norm(x,p=float('inf'))
    
    sgn_x=((x>0).float()-0.5)*2
    
    p=torch.div(torch.abs(x),x_norm)
    renormalize_p=torch.mul(p,n)
    floor_p=torch.floor(renormalize_p)
    compare=torch.rand_like(floor_p)
    final_p=renormalize_p-floor_p
    margin=(compare < final_p).float()
    xi=(floor_p+margin)/n
    
    
    
    Tilde_x=x_norm*sgn_x*xi
    
    return Tilde_x

def select_gpu():
    # Check the availability and utilization of GPUs
    gpu_devices = torch.cuda.device_count()
    gpu_utilization = [torch.cuda.memory_allocated(device) for device in range(gpu_devices)]

    # Select the GPU with the highest available memory
    device = torch.device(f"cuda:{gpu_utilization.index(min(gpu_utilization))}" if torch.cuda.is_available() else "cpu")
    return device
def print_class_distribution(dataset):
    """
    统计并打印数据集中每个类别的占比。
    """
    targets = torch.tensor([label for _, label in dataset])
    class_counts = Counter(targets.tolist())
    total_samples = len(dataset)

    print("Class Distribution:")
    for class_id, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  - Class {class_id}: {count} samples ({percentage:.2f}%)")
# def print_class_distribution(dataset):
#     if isinstance(dataset, torch.utils.data.Subset):
#         targets = [dataset.dataset.targets[i].item() if isinstance(dataset.dataset.targets[i], torch.Tensor) else dataset.dataset.targets[i] for i in dataset.indices]
#     else:
#         targets = [t.item() if isinstance(t, torch.Tensor) else t for t in dataset.targets]
#     counter = collections.Counter(targets)
#     total = len(targets)
#     for class_id, count in counter.items():
#         percentage = count / total * 100
#         print(f"Class {class_id}: {percentage:.2f}%")


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError


class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), torch.float32
    
from torch.utils.data import Dataset
import json
import math
import sys
class MyDataSet(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, item):
        input = self.inputs[item]
        target = self.targets[item]

        return input, target

    def __len__(self):
        return len(self.inputs)
def generate_vocab(data_path, vocab_path='./vocab.json'):
    word_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list += sentence.split()
    word_list = list(set(word_list))

    # 生成字典
    word2index_dict = {w: i + 2 for i, w in enumerate(word_list)}
    word2index_dict['<PAD>'] = 0
    word2index_dict['<UNK>'] = 1
    word2index_dict = dict(sorted(word2index_dict.items(), key=lambda x: x[1]))  # 排序

    index2word_dict = {index: word for word, index in word2index_dict.items()}

    # 将单词表写入json
    json_str = json.dumps(word2index_dict, indent=4)
    with open(vocab_path, 'w') as json_file:
        json_file.write(json_str)

    return word2index_dict, index2word_dict


def generate_dataset(data_path, word2index_dict, n_step=5):
    """
    :param data_path: 数据集路径
    :param word2index_dict: word2index字典
    :param n_step: 窗口大小
    :return: 实例化后的数据集
    """
    def word2index(word):
        try:
            return word2index_dict[word]
        except:
            return 1  # <UNK>

    input_list = []
    target_list = []

    f = open(data_path, 'r')
    lines = f.readlines()
    for sentence in lines:
        word_list = sentence.split()
        if len(word_list) < n_step + 1:  # 句子中单词不足，padding
            word_list = ['<PAD>'] * (n_step + 1 - len(word_list)) + word_list
        index_list = [word2index(word) for word in word_list]
        for i in range(len(word_list) - n_step):
            input = index_list[i: i + n_step]
            target = index_list[i + n_step]

            input_list.append(torch.tensor(input))
            target_list.append(torch.tensor(target))

    # 实例化数据集
    dataset = MyDataSet(input_list, target_list)

    return dataset


def train_one_epoch(model, loss_function, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    optimizer.zero_grad()

    # RNNLM based on Attention
    # attention_epoch_path = os.path.join(attention_path, f'{str(epoch).zfill(3)}epoch')
    # os.makedirs(attention_epoch_path)

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data

        pred = model(input.to(device))

        # RNNLM based on Attention
        # pred, attention = model(input.to(device))
        # img = attention.cpu().numpy() * 255
        # img = img.astype(np.uint8)
        # im_color = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        # cv2.imwrite(os.path.join(attention_epoch_path, f'{step}.png'), im_color)

        loss = loss_function(pred, target.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, ppl: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            math.exp(accu_loss.item() / (step + 1)),
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        # gradient clip
        # clip_grad_norm_(parameters=model.parameters(), max_norm=0.1, norm_type=2)
        optimizer.step()
        optimizer.zero_grad()
        # update lr
        if lr_scheduler != None:
            lr_scheduler.step()

    return accu_loss.item() / (step + 1), math.exp(accu_loss.item() / (step + 1))


def evaluate(model, loss_function, data_loader, device, epoch):
    model.eval()

    accu_loss = torch.zeros(1).to(device)  # 累计损失

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        input, target = data

        pred = model(input.to(device))

        # RNNLM based on Attention
        # pred, _ = model(input.to(device))

        loss = loss_function(pred, target.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, ppl: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            math.exp(accu_loss.item() / (step + 1)),
        )

    return accu_loss.item() / (step + 1), math.exp(accu_loss.item() / (step + 1))


# 调度器，Poly策略
def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

class RunningMeanStd(object):
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = (), device="cpu"):
        """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        """
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon

    def update(self, arr):
        arr = arr.reshape(-1, arr.size(-1))
        batch_mean = torch.mean(arr, dim=0)
        batch_var = torch.var(arr, dim=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count: int):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = (
            m_a
            + m_b
            + torch.square(delta)
            * self.count
            * batch_count
            / (self.count + batch_count)
        )
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True