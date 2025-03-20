import collections
import numpy as np
import torch
import random
import torch.nn as nn
from typing import Tuple
from collections import Counter
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

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