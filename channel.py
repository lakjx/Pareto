import torch
import numpy as np
# def simulate_rayleigh_channel():
#     # 创建两个指数分布的随机变量
#     exp1 = torch.distributions.Exponential(1.25)
#     exp2 = torch.distributions.Exponential(0.85)

#     # 生成随机数
#     x = exp1.sample()
#     y = exp2.sample()

#     # 计算平方和的平方根
#     h = torch.sqrt(x**2 + y**2)

#     return h
# 瑞利衰落信道
def simulate_rayleigh_channel(N0,B,P_tx,PL = 100): # 路径损耗因子,单位为 dB
    h = np.sqrt(0.01) * (np.random.normal(0, 1, 100) + 1j * np.random.normal(0, 1, 100))
    # # 计算信噪比
    # SNR = P_tx / (N0 * B)
    # 计算信道容量
    # C = B * np.log2(1 + SNR)
    SINR = P_tx * np.abs(h)**2 / (10**(PL/10) * N0 * B)
    throughput = B * np.mean(np.log2(1 + SINR))
    return throughput


if __name__ == '__main__':
    h = simulate_rayleigh_channel()
    print(h)
