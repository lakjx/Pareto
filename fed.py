import time
import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import torch.multiprocessing as mp
from multiprocessing import Pool
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils import quantize,print_class_distribution,weights_init,select_gpu,set_random_seed
from channel import simulate_rayleigh_channel
import datetime
from functools import partial

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
# 定义联邦学习客户端
class FLClient(nn.Module):
    def __init__(self, client_id, dataset_name,cpu_freq=0,transmission_power=0):
        super(FLClient, self).__init__()
        self.client_id = client_id
        self.dataset_name = dataset_name
        self.dataset = None
        self.quantization_bit = 32
        self.cpu_freq = cpu_freq if cpu_freq != 0 else np.random.uniform(0.5, 3.5)*1e9
        self.transmission_power = transmission_power if transmission_power != 0 else np.random.uniform(10, 2000)*1e-3
        # 定义神经网络结构
        if dataset_name == 'MNIST':
            self.model = nn.Sequential(
                Flatten(),
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        elif dataset_name == 'FashionMNIST':
            self.model = nn.Sequential(
                nn.Conv2d(1, 10, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(10, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                Flatten(),
                nn.Linear(320, 50),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(50, 10)
            )
        elif dataset_name == 'CIFAR10':
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=48, kernel_size=(3,3), padding=(1,1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3,3), padding=(1,1)),
                nn.ReLU(),
                nn.MaxPool2d(2,2),                
                nn.Dropout(0.2),
                nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3,3), padding=(1,1)),
                nn.ReLU(),
                nn.Conv2d(in_channels=192, out_channels=256, kernel_size=(3,3), padding=(1,1)),
                nn.MaxPool2d(2,2),
                nn.Dropout(0.2),
                Flatten(),
                nn.Linear(in_features=8*8*256, out_features=512),
                nn.ReLU(),
                nn.Linear(in_features=512, out_features=64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(in_features=64, out_features=10)
            )
            # self.model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=10)
        # 计算模型参数的大小
        self.model_size = sum(p.numel() for p in self.model.parameters()) * 4  # float类型的字节大小为4
        # 将模型加载到gpu上
        self.to_device()
        # 初始化模型参数
        self.model.apply(weights_init)
    def to_device(self):
        # 选择第0块GPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    def forward(self, x):
        return self.model(x)
    def load_data(self,data):
        self.dataset = data
        # 计算数据集的大小，单位：字节
        num_elements = torch.prod(torch.tensor(self.dataset[0][0].size()))
        element_size = self.dataset[0][0].element_size()
        self.datasetsize_Byte = num_elements.item() * element_size
        # 计算数据集数量
        self.data_num = len(self.dataset)

    def calculate_local_computation_delay(self, data_size, epochs,cpu_cycles_per_bit=988):
        # 计算本地计算时延，单位：s
        return 8*data_size * epochs* cpu_cycles_per_bit/ self.cpu_freq
    def calculate_upload_delay(self, bandwidth,N0=4*10e-21):
        # 计算上传时延，单位：s
        # 使用香农公式计算传输速率  bandwidth,单位：Hz
        transmission_rate = simulate_rayleigh_channel(N0,bandwidth,self.transmission_power)
        upload_delay = 8*self.model_size / transmission_rate/32*self.quantization_bit
        self.com_overhead = self.model_size / 32*self.quantization_bit/1e6  #bytes ->MB
        return upload_delay
    def calculate_local_computation_energy(self, data_size, epochs,cpu_cycles_per_bit=988,alpha=2e-28):
        # 计算本地计算能耗，单位：J
        return 8*data_size * cpu_cycles_per_bit * (self.cpu_freq ** 2)*epochs*alpha
    def calculate_upload_energy(self, upload_delay):
        # 计算上传能耗，单位：J
        return self.transmission_power * upload_delay
    def local_train(self,global_model, epochs, batch_size,bandwidth=10e6):
            # 但是在训练之前,我们需要将全局模型的参数复制到本地模型
            self.load_state_dict(global_model.state_dict())
            self.model.train()
            dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

            # 在这里进行客户端的本地训练
            optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()

            # 记录每个epoch的损失和准确率
            losses = []
            accuracies = []

            for epoch in range(epochs):
                epoch_loss = 0
                correct = 0
                for data, target in dataloader:
                    data, target = data.to(self.device), target.to(self.device)
                    optimizer.zero_grad()
                    output = self.forward(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    # 计算损失和准确率
                    epoch_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    correct += (predicted == target).sum().item()
                losses.append(epoch_loss / len(dataloader))
                accuracies.append(correct / len(dataloader.dataset))
                print(f"Client ID: {self.client_id}, Local Epochs: {epoch+1}/{epochs}")
                print(f"Client Loss: {losses[-1]}, Client Accuracy: {accuracies[-1]}")
            
            #计算本地计算时延
            self.local_computation_delay = self.calculate_local_computation_delay(self.datasetsize_Byte, epochs)
            #计算本地计算能耗
            self.local_computation_energy = self.calculate_local_computation_energy(self.datasetsize_Byte, epochs)
            #计算上传时延
            self.upload_delay = self.calculate_upload_delay(bandwidth)
            #计算上传能耗
            self.upload_energy = self.calculate_upload_energy(self.upload_delay)
            
            return losses, accuracies

# 定义联邦学习服务器
class FLServer:
    def __init__(self, num_clients, dataset_names,Noniid_level=1,reload=False,client = FLClient):
        self.num_clients = num_clients
        self.clients = [client(i, dataset_names) for i in range(num_clients)]
        self.global_model = client("Server",dataset_names)  # 创建全局模型
        self.avg_losses = []
        self.avg_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        self.time_lapse = []
        self.energy_consume = []
        self.com_overheads = []
        if dataset_names == 'MNIST':
            self.dataset = datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        elif dataset_names == 'FashionMNIST':
            self.dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        elif dataset_names == 'CIFAR10':
            self.dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
        self.distribute_data(non_iid_level=Noniid_level)
        if reload:
            self.global_model.load_state_dict(torch.load(f'./checkpoint/global_model_{dataset_names}.pth'))
    # def distribute_data(self, non_iid_level=0.3):
    #     """将数据集以 non-iid 方式分配给客户端。"""

    #     # 获取数据集中所有类别
    #     if isinstance(self.dataset.targets, torch.Tensor):
    #         classes = torch.unique(self.dataset.targets)
    #     elif isinstance(self.dataset.targets, list):
    #         classes = torch.unique(torch.tensor(self.dataset.targets))
    #     num_classes = len(classes)

    #     # 为每个类别创建索引列表
    #     class_indices = [[] for _ in range(num_classes)]
    #     for i, label in enumerate(self.dataset.targets):
    #         class_indices[label].append(i)

    #     # 为每个客户端分配数据
    #     for i, client in enumerate(self.clients):
    #         # 随机选择类别数量
    #         num_client_classes = max(1, int(round(num_classes * non_iid_level)))
    #         chosen_classes = np.random.choice(num_classes, num_client_classes, replace=False)

    #         # 收集选定类别的数据索引
    #         client_indices = []
    #         for class_idx in chosen_classes:
    #             client_indices.extend(class_indices[class_idx])

    #         # 创建客户端数据集
    #         client_dataset = torch.utils.data.Subset(self.dataset, client_indices)
    #         client.load_data(client_dataset)
    #         print(f"Client {i} has {len(client_dataset)} samples")
    #         print_class_distribution(client_dataset) 
    def distribute_data(self,non_iid_level=1):
        # 将数据集分配给客户端
        data_size = len(self.dataset) // self.num_clients
        data_indices = list(range(len(self.dataset)))
        np.random.shuffle(data_indices)

        #获取数据集中所有类别
        classes = np.unique(self.dataset.targets)
        num_classes = len(classes)

        # 计算每个客户端应该有多少个类别
        num_classes_per_client = int(num_classes * non_iid_level)

        for i, client in enumerate(self.clients):
            start = i * data_size
            end = start + data_size

            # 随机选择一些类别
            chosen_classes = torch.randperm(num_classes)[:num_classes_per_client].clone().detach()
            # 只选择这些类别的数据
            client_indices = [idx for idx in data_indices[start:end] if self.dataset.targets[idx] in chosen_classes]
            client_dataset = torch.utils.data.Subset(self.dataset, client_indices)
            client.load_data(client_dataset)
            print(f"Client {i} has {len(client_dataset)} samples")
            print_class_distribution(client_dataset)

    def train(self, num_rounds, num_participating, local_epochs, local_batch_size,quantization_bit,clients_cpu_fre=[0.5,0.5,0.5,0.5,0.5],bandwidth=10e6,AdaQuantFL_enable=False):

        clients_cpu_fre=np.multiply(clients_cpu_fre,1e9) #GHz->Hz
        bandwidth=bandwidth*1e6 #MHz->Hz
        quan_bit_avg = []
        # 训练全局模型
        for round in range(num_rounds):  
            # 随机选择参与本轮训练的客户端
            # num_participating = min(num_participating, self.num_clients)
            participating_clients = np.random.choice(self.num_clients, num_participating, replace=False)
            # 设置每个客户端的量化位数
            for ii in participating_clients:
                self.clients[ii].quantization_bit = quantization_bit[ii]
                self.clients[ii].cpu_freq = clients_cpu_fre[ii]
                quan_bit_avg.append(quantization_bit[ii])
            
            # 在这里进行客户端的本地训练
            round_losses = []
            round_accuracies = []
            round_timelapse = []
            round_energyconsume = []
            communication_cost = 0
            t0 = time.time()
            for client_id in participating_clients:
                self.clients[client_id].to_device()
                losses,accuracies = self.clients[client_id].local_train(self.global_model, local_epochs, local_batch_size,bandwidth)
                round_losses.extend(losses)
                round_accuracies.extend(accuracies)
                round_timelapse.append(self.clients[client_id].local_computation_delay + self.clients[client_id].upload_delay)
                round_energyconsume.append(self.clients[client_id].local_computation_energy + self.clients[client_id].upload_energy)
                communication_cost += self.clients[client_id].com_overhead
            t1 = time.time()
            print(f"not multipro Time: {t1-t0}")

            # 计算并记录本轮的平均准确率和损失
            self.avg_losses.append(sum(round_losses) / len(round_losses))
            self.avg_accuracies.append(sum(round_accuracies) / len(round_accuracies))
            # 计算并记录本轮的平均时延和能耗
            self.time_lapse.append(max(round_timelapse))
            self.energy_consume.append(sum(round_energyconsume))
            self.com_overheads.append(communication_cost)
            
            # 在这里进行联邦学习的聚合
            self.aggregate_models(participating_clients)
            true_iter = len(self.avg_losses)
            
            # 评估全局模型
            test_loss, test_accuracy = self.evaluate_global_model(self.global_model,device = self.global_model.device)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_accuracy)
            
            if true_iter % 1 == 0:
                print(f"-----------------Global Model Evaluation of {self.global_model.dataset_name}---------------")
                print(f"local_epochs: {local_epochs}, local_batch_size: {local_batch_size}, quantization_bit: {quantization_bit}")
                print(f"Round: {true_iter}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Communication Cost(MByte): {communication_cost}")
                print("---------------------------------------------------------")
            #AdaQuantFL
            if AdaQuantFL_enable:
                if round == 0:
                    F0 = self.avg_losses[-1] * 1
                else:
                    Fk = self.avg_losses[-1] * 1
                    s_k = np.round(np.sqrt(F0/Fk) * quantization_bit[0])
                    quantization_bit = np.clip([s_k]*5,1,32)  
            terminate = False
        episode_result = np.stack((np.array(self.avg_losses[-1]), np.array(self.avg_accuracies[-1]), np.array(self.test_losses[-1]), np.array(self.test_accuracies[-1]), np.array(self.time_lapse[-1]), np.array(self.energy_consume[-1])))  
        return true_iter,np.mean(quan_bit_avg),communication_cost,episode_result

    # quantized_params = [quantize(param.detach(), {'n': client.quantization_bit}) for param, client_id in zip(client_params, participating_clients) for client in self.clients if client.client_id == client_id]
    def aggregate_models(self, participating_clients):
        # 计算平均模型参数
        with torch.no_grad():
            total_data_num = sum(self.clients[client_id].data_num for client_id in participating_clients)
            total_quantization_bit = sum(self.clients[client_id].quantization_bit for client_id in participating_clients)
            for global_param, *client_params in zip(self.global_model.parameters(), *[self.clients[client_id].model.parameters() for client_id in participating_clients]):
                quantized_params = []
                for param, client_id in zip(client_params, participating_clients):
                    for client in self.clients:
                        if client.client_id == client_id:
                            quantized_param = quantize(param.detach(), {'n': client.quantization_bit})
                            # quantized_param = param
                            quantized_params.append(quantized_param)
                weighted_params = [quantized_param * (self.clients[client_id].data_num / total_data_num) for quantized_param, client_id in zip(quantized_params, participating_clients)]
                param_sum = torch.stack(weighted_params, dim=0).sum(dim=0)
                global_param.copy_(param_sum)

    def evaluate_global_model(self,global_model,device):
                # 加载测试集
            if self.global_model.dataset_name == 'MNIST':
                test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())
            elif self.global_model.dataset_name == 'FashionMNIST':
                test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transforms.ToTensor())
            elif self.global_model.dataset_name == 'CIFAR10':
                test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
           
            global_model.to(device)
            global_model.eval()
            test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            criterion = nn.CrossEntropyLoss()

            test_loss = 0
            test_correct = 0
            with torch.no_grad():
                for data, target in test_dataloader:
                    data, target = data.to(device), target.to(device)
                    output = global_model(data)
                    test_loss += criterion(output, target).item()
                    _, predicted = torch.max(output, 1)
                    test_correct += (predicted == target).sum().item()

            test_loss /= len(test_dataloader)
            test_accuracy = test_correct / len(test_dataset)

            return test_loss, test_accuracy
    def save_metrics_to_excel(self, excel_dir,band=[10,10,10]):
        # 如果路径不存在，则创建路径
        if not os.path.exists(excel_dir):
            os.makedirs(excel_dir)

        # 创建一个字典，其中的键是列名，值是数据
        data = {
            'Round': range(len(self.avg_losses)),
            #记录测试损失和准确率
            'Loss ': self.test_losses,
            'Accuracy': self.test_accuracies,
            'Overheads': self.com_overheads,
            'Energy': self.energy_consume,
            'Delay': self.time_lapse,
            'bw1':band[0],
            'bw2':band[1],
            'bw3':band[2],
        }

        # 创建一个DataFrame
        df = pd.DataFrame(data)

        # 保存到excel_dir路径文件夹下
        df.to_excel(f'{excel_dir}/'+f'{self.global_model.dataset_name}.xlsx', index=False)

class FLClient_Prox(FLClient):
    def __init__(self, client_id, dataset_name,cpu_freq=0,transmission_power=0):
        super(FLClient_Prox, self).__init__(client_id, dataset_name,cpu_freq,transmission_power)
        self.mu = 0.1

    def local_train(self,global_model, epochs, batch_size,bandwidth=10e6):
        self.load_state_dict(global_model.state_dict())
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)

                #FedProx
                prox_term = 0
                for w, w_glob in zip(self.parameters(), global_model.parameters()):
                    prox_term += (self.mu / 2) * torch.norm(w - w_glob) ** 2
                loss += prox_term

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
            losses.append(epoch_loss / len(dataloader))
            accuracies.append(correct / len(dataloader.dataset))
            print(f"Client ID: {self.client_id}, Local Epochs: {epoch+1}/{epochs}")
            print(f"Client Loss: {losses[-1]}, Client Accuracy: {accuracies[-1]}")
        
        self.local_computation_delay = self.calculate_local_computation_delay(self.datasetsize_Byte, epochs)
        self.local_computation_energy = self.calculate_local_computation_energy(self.datasetsize_Byte, epochs)
        self.upload_delay = self.calculate_upload_delay(bandwidth)
        self.upload_energy = self.calculate_upload_energy(self.upload_delay)

        return losses, accuracies

class FLClient_FedDQ(FLClient):
    def __init__(self, client_id, dataset_name,cpu_freq=0,transmission_power=0):
        super(FLClient_FedDQ, self).__init__(client_id, dataset_name,cpu_freq,transmission_power)
        self.resolution = 0.01

    def local_train(self,global_model, epochs, batch_size,bandwidth=10e6):
        self.load_state_dict(global_model.state_dict())
        self.model.train()
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        losses = []
        accuracies = []

        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.forward(data)
                loss = criterion(output, target)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()
            losses.append(epoch_loss / len(dataloader))
            accuracies.append(correct / len(dataloader.dataset))
            print(f"Client ID: {self.client_id}, Local Epochs: {epoch+1}/{epochs}")
            print(f"Client Loss: {losses[-1]}, Client Accuracy: {accuracies[-1]}")
        
        #FedDQ
        bitsnode = 0
        for name, param in self.named_parameters():
            diff_w = (param - global_model.state_dict()[name]).detach().cpu().numpy()
            rang_diff = np.max(diff_w) - np.min(diff_w)
            bitsnode += max(np.ceil(np.log2(rang_diff/self.resolution)),2)
        self.quantization_bit = np.round(bitsnode/len(list(self.model.parameters())))

        self.local_computation_delay = self.calculate_local_computation_delay(self.datasetsize_Byte, epochs)
        self.local_computation_energy = self.calculate_local_computation_energy(self.datasetsize_Byte, epochs)
        self.upload_delay = self.calculate_upload_delay(bandwidth)
        self.upload_energy = self.calculate_upload_energy(self.upload_delay)

        return losses, accuracies

if __name__ == '__main__':
    set_random_seed(1568)
    dataset_names = ['MNIST', 'FashionMNIST', 'CIFAR10']
    num_participating = [3, 3, 3]
    local_epochs = [3, 3, 3]
    batch_size = [128, 128, 128]
    # quantization_bit = [[2,2,2,2,2],[2,2,2,2,2],[2,2,2,2,2]]
    quantization_bit = [[8,8,8,8,8],[8,8,8,8,8],[8,8,8,8,8]]
    cpu_freq = [[0.5,0.6,0.7,1,0.5],[0.5,0.8,0.6,0.7,0.5],[0.5,0.8,1.0,0.7,0.5]]
    BW = [10e6,10e6,10e6]

    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    # FL_env = [FLServer(5, dataset_name,client=FLClient_Prox) for dataset_name in dataset_names]
    FL_env = [FLServer(5, dataset_name,client=FLClient_FedDQ) for dataset_name in dataset_names]
    sav_dir = './results/FedDQ/'
    for id in range(3):
        FL_env[id].train(40,3,3,256,quantization_bit[id],cpu_freq[id],BW[id],AdaQuantFL_enable=False)
        FL_env[id].save_metrics_to_excel(sav_dir)
