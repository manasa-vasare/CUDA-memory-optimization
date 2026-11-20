import os
import matplotlib.pyplot as plt
import numpy as np

# Set styling
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

output_dir = os.path.join(os.path.dirname(__file__), '..', 'assets', 'charts')
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------------------
# GRAPH 1: Transpose Execution Time & Bandwidth Comparison
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

runs = ['Run 1', 'Run 2', 'Run 3', 'Mean']
uncoalesced_time = [0.797, 0.796, 0.809, 0.801]
coalesced_time = [0.417, 0.418, 0.418, 0.418]

x = np.arange(len(runs))
width = 0.35

# Subplot 1: Execution Time
rects1 = ax1.bar(x - width/2, uncoalesced_time, width, label='Uncoalesced (Naive)', color='#E74C3C', edgecolor='#C0392B', alpha=0.9)
rects2 = ax1.bar(x + width/2, coalesced_time, width, label='Coalesced (Shared Mem)', color='#2ECC71', edgecolor='#27AE60', alpha=0.9)

ax1.set_title('Matrix Transpose Execution Time ($N=4096$)\nLower is Better', fontsize=13, fontweight='bold', pad=15)
ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='semibold')
ax1.set_xticks(x)
ax1.set_xticklabels(runs, fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#E0E0E0', fontsize=10)

# Value labels on bars
for bar in rects1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#C0392B')

for bar in rects2:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f'{yval:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1E8449')

# Speedup annotation on Mean
ax1.annotate('1.92x Speedup\n(47.8% Time Reduction)', 
             xy=(3 + width/2, 0.418), 
             xytext=(2.2, 0.70),
             arrowprops=dict(facecolor='#2980B9', edgecolor='#2980B9', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9.5, fontweight='bold', color='#2980B9',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1'))

# Subplot 2: Effective Bandwidth
uncoalesced_bw = [168.4, 168.7, 165.8, 167.6]
coalesced_bw = [322.1, 321.5, 320.8, 321.5]

rects3 = ax2.bar(x - width/2, uncoalesced_bw, width, label='Uncoalesced (Naive)', color='#E67E22', edgecolor='#D35400', alpha=0.9)
rects4 = ax2.bar(x + width/2, coalesced_bw, width, label='Coalesced (Shared Mem)', color='#3498DB', edgecolor='#2980B9', alpha=0.9)

ax2.set_title('Effective Memory Bandwidth ($N=4096$)\nHigher is Better', fontsize=13, fontweight='bold', pad=15)
ax2.set_ylabel('Bandwidth (GB/s)', fontsize=11, fontweight='semibold')
ax2.set_xticks(x)
ax2.set_xticklabels(runs, fontsize=10)
ax2.set_ylim(0, 400)
ax2.grid(axis='y', linestyle='--', alpha=0.5)
ax2.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#E0E0E0', fontsize=10)

for bar in rects3:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#BA4A00')

for bar in rects4:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 5, f'{yval:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#1F618D')

# Bandwidth scaling annotation
ax2.annotate('+91.8% Bandwidth\n(Nearly Doubled)', 
             xy=(3 + width/2, 321.5), 
             xytext=(1.8, 350),
             arrowprops=dict(facecolor='#16A085', edgecolor='#16A085', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9.5, fontweight='bold', color='#16A085',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8F5', edgecolor='#A3E4D7'))

plt.suptitle('NVIDIA GeForce RTX 5060 Ti: Coalesced vs. Uncoalesced Matrix Transpose Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
chart1_path = os.path.join(output_dir, 'transpose_performance_comparison.png')
plt.savefig(chart1_path, bbox_inches='tight')
plt.close()
print(f"Generated: {chart1_path}")


# -------------------------------------------------------------
# GRAPH 2: Reference Baseline vs RTX 5060 Ti Comparison
# -------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

benchmarks = ['Uncoalesced Transpose\n(N=4096)', 'Coalesced Transpose\n(N=4096)', 'Naive MatMul\n(N=1024)']
t4_times = [1.850, 0.720, 3.900]
rtx5060_times = [0.797, 0.417, 1.399]

x_bm = np.arange(len(benchmarks))
width = 0.35

b1 = ax1.bar(x_bm - width/2, t4_times, width, label='Tesla T4 (CC 7.5 Reference)', color='#7F8C8D', edgecolor='#566573', alpha=0.9)
b2 = ax1.bar(x_bm + width/2, rtx5060_times, width, label='RTX 5060 Ti (CC 12.0 Observed)', color='#76B900', edgecolor='#5B8F00', alpha=0.95)

ax1.set_title('Kernel Execution Time: Tesla T4 vs. RTX 5060 Ti\nLower is Better', fontsize=12, fontweight='bold', pad=12)
ax1.set_ylabel('Execution Time (ms)', fontsize=11, fontweight='semibold')
ax1.set_xticks(x_bm)
ax1.set_xticklabels(benchmarks, fontsize=9.5)
ax1.set_ylim(0, 4.5)
ax1.grid(axis='y', linestyle='--', alpha=0.5)
ax1.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#E0E0E0', fontsize=10)

for bar in b1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.08, f'{y:.3f} ms', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#566573')

for bar in b2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 0.08, f'{y:.3f} ms', ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#386300')

# Speedup multipliers
speedups = ['2.32x Faster', '1.73x Faster', '2.79x Faster']
for i, s in enumerate(speedups):
    ax1.text(x_bm[i] + width/2, rtx5060_times[i] + 0.45, s, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#E8F5E9', edgecolor='#C8E6C9'))

# Subplot 2: Compute Throughput (GFLOPS)
devices = ['Tesla T4\n(CC 7.5 Reference)', 'RTX 5060 Ti\n(CC 12.0 Observed)']
gflops = [550.6, 1534.9]
colors = ['#7F8C8D', '#76B900']

b_gflops = ax2.bar(devices, gflops, width=0.45, color=colors, edgecolor=['#566573', '#5B8F00'], alpha=0.95)
ax2.set_title('Naive Matrix Multiplication Compute Rate ($N=1024$)\nHigher is Better', fontsize=12, fontweight='bold', pad=12)
ax2.set_ylabel('Throughput (GFLOPS)', fontsize=11, fontweight='semibold')
ax2.set_ylim(0, 1850)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

for bar in b_gflops:
    y = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2.0, y + 30, f'{y:.1f} GFLOPS', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2C3E50')

ax2.annotate('2.79x Higher Throughput\n(+984.3 GFLOPS)', 
             xy=(1, 1534.9), 
             xytext=(0.35, 1300),
             arrowprops=dict(facecolor='#2980B9', edgecolor='#2980B9', shrink=0.08, width=1.5, headwidth=7),
             fontsize=9.5, fontweight='bold', color='#2980B9',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1'))

plt.suptitle('Hardware Generation Performance Scaling: Tesla T4 (Turing) vs. RTX 5060 Ti (Blackwell)', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
chart2_path = os.path.join(output_dir, 'hardware_comparison_t4_vs_rtx5060ti.png')
plt.savefig(chart2_path, bbox_inches='tight')
plt.close()
print(f"Generated: {chart2_path}")


# -------------------------------------------------------------
# GRAPH 3: Multi-Run Stability for Transpose & MatMul
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.8), dpi=300)
fig.patch.set_facecolor('#FFFFFF')

runs_x = np.array([1, 2, 3])
uncoalesced_times = np.array([0.797, 0.796, 0.809])
coalesced_times = np.array([0.417, 0.418, 0.418])
matmul_times = np.array([1.399, 1.400, 1.511])

ax.plot(runs_x, uncoalesced_times, marker='o', linewidth=2.5, markersize=8, label='Uncoalesced Transpose (N=4096)', color='#E74C3C')
ax.plot(runs_x, coalesced_times, marker='s', linewidth=2.5, markersize=8, label='Coalesced Transpose (N=4096)', color='#2ECC71')
ax.plot(runs_x, matmul_times, marker='^', linewidth=2.5, markersize=8, label='Naive MatMul (N=1024)', color='#9B59B6')

ax.set_title('Multi-Run Execution Time Stability on NVIDIA RTX 5060 Ti', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Benchmark Execution Run', fontsize=11, fontweight='semibold')
ax.set_ylabel('Time (ms)', fontsize=11, fontweight='semibold')
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(['Run 1 (10:15:39 AM)', 'Run 2 (10:15:43 AM)', 'Run 3 (10:16:10 AM)'], fontsize=10)
ax.set_ylim(0.2, 1.7)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(frameon=True, facecolor='#F8F9FA', edgecolor='#E0E0E0', fontsize=10, loc='center right')

# Annotations of variance
for i in range(3):
    ax.text(runs_x[i], uncoalesced_times[i] + 0.04, f'{uncoalesced_times[i]:.3f} ms', ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')
    ax.text(runs_x[i], coalesced_times[i] - 0.07, f'{coalesced_times[i]:.3f} ms', ha='center', fontsize=8.5, fontweight='bold', color='#1E8449')
    ax.text(runs_x[i], matmul_times[i] + 0.04, f'{matmul_times[i]:.3f} ms', ha='center', fontsize=8.5, fontweight='bold', color='#8E44AD')

plt.tight_layout()
chart3_path = os.path.join(output_dir, 'multi_run_stability.png')
plt.savefig(chart3_path, bbox_inches='tight')
plt.close()
print(f"Generated: {chart3_path}")
