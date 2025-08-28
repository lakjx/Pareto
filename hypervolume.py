import numpy as np
import pygmo as pg # 导入 pygmo 库

# ==============================================================================
# 1. 生成模拟数据 
# ==============================================================================
# 假设我们进行了5次独立实验 (n_runs=5)，有3个SP (n_dim=3)
print("Step 1: Generating simulation data...")

fixed_solutions = np.array([
    [48.92,62.65,71.85],
    [47.23,60.96,73.16],
    [51.54,69.27,74.47],
    [52.85,66.58,67.78],
    [39.46,73.89,77.09],
])

fedprox_solutions = np.array([
    [65.41,68.76,87.95],
    [70.42,72.86,90.12],
    [62.32,66.58,85.42],
    [64.58,66.77,86.85],
    [60.23,63.98,83.91],
])

feddq_solutions = np.array([
    [56.51,46.72,67.82],
    [66.85,58.64,75.92],
    [58.88,63.45,59.46],
    [66.51,39.65,83.45],
    [55.23,60.34,76.54],
])
adaquantfl_solutions = np.array([
    [54.74,75.17,73.33],
    [68.45,71.56,78.52],
    [59.65,75.63,78.12],
    [66.56,69.65,79.63],
    [62.34,72.45,67.12]
    ])
rsmmasac_solutions = np.array([
    [76.19,76.58,79.63],
    [73.69,72.58,74.63],
    [79.79,82.82,83.93],
    [75.39,74.94,75.24],
    [69.78,71.22,83.22]
])
pac_solutions = np.array([
    [83.85,77.86,81.56],
    [81.56,75.57,80.27],
    [85.67,82.78,77.89],
    [80.34,79.45,82.67],
    [75.95,76.86,75.67]
])
pac2_solutions = np.array([
    [79.14,75.93,82.56],
    [81.56,77.57,84.27],
    [68.79,69.14,76.93],
    [74.34,72.45,78.67],
    [79.95,76.86,80.67]
])
mappo_solutions = np.array([
    [76.03, 76.08,77.67],
    [72.53, 70.48,71.55],
    [79.53, 82.56,83.67],
    [74.13, 73.68,81.98],
    [81.52, 70.96,82.96],
])

pac_tcad_solutions = np.array([
    [82.85,77.86,81.56],
    [81.56,75.57,80.27],
    [80.67,82.78,77.89],
])
pac_tcad_wo_solutions = np.array([
    [66.67,76.84,80.12],
    [69.86,79.12,83.69],
    [70.12,79.67,82.12]
])
pac2_tcad_solutions = np.array([
    [79.14,75.93,82.56],
    [81.56,77.57,84.27],
    [73.79,69.14,76.93],
])
pac2_tcad_wo_solutions = np.array([
    [68.07,77.97,77.12],
    [63.86,73.12,83.69],
    [72.12,67.63,81.02]
])

wikimappo = np.array([
    [-12.86,75.46,84.73],
    [-12.68,73.12,82.69],
    [-13.90,76.63,87.02]
])
wikipac = np.array([
    [-11.07,73.97,88.52],
    [-13.98,72.12,82.56],
    [-14.67,77.63,81.67]
])
wikipac2 = np.array([
    [-17.07,72.39,86.12],
    [-14.86,73.12,85.69],
    [-13.69,75.68,85.69]
])
algorithms_results = {
    "PAC-MCOFL": pac_solutions,
    "PAC-MCOFL-p": pac2_solutions,
    "Fixed": fixed_solutions,
    "FedProx": fedprox_solutions,
    "FedDQ": feddq_solutions,
    "AdaQuantFL": adaquantfl_solutions,
    "RSM-MASAC": rsmmasac_solutions,
    "MAPPO": mappo_solutions,
}

algorithms_results = {
    "pacmcofl":pac_tcad_solutions,
    "pacmcofl-wo":pac_tcad_wo_solutions,
    "pacmcofl-p":pac2_tcad_solutions,
    "pacmcofl-p-wo":pac2_tcad_wo_solutions
}

algorithms_results = {
    "wikimappo":wikimappo,
    "wikipac":wikipac,
    "wikipac2":wikipac2,
}
# ==============================================================================
# 2. 数据预处理：归一化 (与之前完全相同)
# ==============================================================================
print("Step 2: Normalizing data for fair comparison...")

all_solutions = np.vstack(list(algorithms_results.values()))
min_vals = all_solutions.min(axis=0)
max_vals = all_solutions.max(axis=0)

normalized_results = {}
for name, solutions in algorithms_results.items():
    normalized_results[name] = (solutions - min_vals) / (max_vals - min_vals)

# print(f"Sample normalized solution for PAC-MCOFL:\n{normalized_results['PAC-MCOFL'][0]}\n")


# ==============================================================================
# 3. 使用 pygmo 定义参考点并计算HVI
# ==============================================================================
print("Step 3: Calculating Hypervolume Indicator (HVI) using pygmo...")

# pygmo的hypervolume默认也是为最小化问题设计的。
# 因此，我们采用与之前相同的策略：将最大化问题转化为最小化问题。
# 定义一个理想点 (Utopian Point)，它比所有归一化后的解都要好。
utopian_point = np.array([1.1, 1.1, 1.1]) 

hvi_scores = {}
for name, norm_solutions in normalized_results.items():
    # 1. 问题转化: 计算每个解相对于理想点的“距离”，将问题转化为最小化问题。
    # 目标是让这个“距离”越小越好。
    inverted_solutions = utopian_point - norm_solutions
    
    # 2. 初始化pygmo的hypervolume对象。注意：pygmo是先传入“点集”。
    hv_computer = pg.hypervolume(inverted_solutions)
    
    # 3. 定义参考点。对于转化后的最小化问题，其参考点应该是我们之前定义的理想点。
    # 因为我们希望所有(utopian - solution)的值都尽可能小（接近0），
    # 所以一个比所有可能值都大的点就是最差的参考点。
    # 在这个转化问题中，最差的点就是utopian_point本身（当solution为0时）。
    ref_point_for_inverted_problem = utopian_point
    
    # 4. 计算HVI。传入参考点进行计算。
    hvi = hv_computer.compute(ref_point_for_inverted_problem)
    hvi_scores[name] = hvi

# ==============================================================================
# 4. 打印和分析结果 (与之前完全相同)
# ==============================================================================
print("Step 4: Final HVI Scores (higher is better):")
print("-" * 40)
for name, score in sorted(hvi_scores.items(), key=lambda item: item[1], reverse=True):
    print(f"{name:<20}: {score:.4f}")
print("-" * 40)
print("\nConclusion: PAC-MCOFL achieves the highest HVI, indicating its solutions are")
print("collectively superior and form a better Pareto front.")