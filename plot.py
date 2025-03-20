import matplotlib.pyplot as plt

import numpy as np


# 数据量

data_volumes = ['500', '600', '700']


# 能耗、时延和通信量

energy_consumption = [50, 55, 60] # 单位：瓦特

time_delay = [10, 15, 20] # 单位：毫秒

communication_volume = [200, 250, 300] # 单位：兆比特


# 创建堆叠柱状图

fig, ax = plt.subplots()


# 绘制能耗

bar1 = ax.bar(data_volumes, energy_consumption, color='b', label='能耗')


# 绘制时延

bar2 = ax.bar(data_volumes, time_delay, bottom=energy_consumption, color='r', label='时延')


# 绘制通信量

bar3 = ax.bar(data_volumes, communication_volume, bottom=np.array(energy_consumption)+np.array(time_delay), color='g', label='通信量')


# 设置图例

ax.legend()


# 添加标题和轴标签

ax.set_title('数据量与能耗、时延和通信量的关系')

ax.set_xlabel('数据量')

ax.set_ylabel('值')


# 显示图表

plt.show()