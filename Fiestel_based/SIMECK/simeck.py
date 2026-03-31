import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA (SIMECK)
# =========================
cpu_time = [
    9.44e-05,
    0.0015057,
    0.0060821,
    0.0272797,
    0.101941,
    0.388793,
    0.974904,
    4.86845,
    9.79886
]

cpu_throughput = [
    0.0867797,
    0.0870505,
    0.0862018,
    0.0768759,
    0.0822886,
    0.0863041,
    0.0860455,
    0.0861528,
    0.085608
]

# =========================
# 🚀 GPU DATA (SIMECK)
# =========================
gpu_kernel_time = [
    0.000155648,
    3.3792e-05,
    8.2624e-05,
    0.000114144,
    0.000311616,
    0.000776,
    0.00156797,
    0.00726029,
    0.0113113
]

gpu_mem_time = [
    4.5088e-05,
    8.0704e-05,
    0.000201504,
    0.000623232,
    0.00185306,
    0.00540685,
    0.0145771,
    0.0641788,
    0.128266
]

gpu_throughput = [
    0.0526316,
    3.87879,
    6.34547,
    18.3729,
    26.9197,
    43.2402,
    53.4999,
    57.7705,
    74.1611
]

# Total GPU time = kernel + memory
gpu_time = [k + m for k, m in zip(gpu_kernel_time, gpu_mem_time)]

# Compute speedup
speedup = [g / c for g, c in zip(gpu_throughput, cpu_throughput)]

# =========================
# 📊 THREE COLUMNS IN ONE FIGURE
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

# --- Time Comparison ---
axes[0].plot(N, cpu_time, marker='o', label='CPU Time', color='blue')
axes[0].plot(N, gpu_time, marker='s', label='GPU Time', color='green')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel("Blocks (N)")
axes[0].set_ylabel("Time (s)")
axes[0].set_title("SIMECK CPU vs GPU Time")
axes[0].grid(True, which='both', ls='--')
axes[0].legend()

# --- Throughput Comparison ---
axes[1].plot(N, cpu_throughput, marker='o', label='CPU Throughput', color='blue')
axes[1].plot(N, gpu_throughput, marker='s', label='GPU Throughput', color='green')
axes[1].set_xscale('log')
axes[1].set_xlabel("Blocks (N)")
axes[1].set_ylabel("Throughput (GB/s)")
axes[1].set_title("SIMECK CPU vs GPU Throughput")
axes[1].grid(True, which='both', ls='--')
axes[1].legend()

# --- Speedup ---
axes[2].plot(N, speedup, marker='^', color='red', label='Speedup (GPU/CPU)')
axes[2].set_xscale('log')
axes[2].set_xlabel("Blocks (N)")
axes[2].set_ylabel("Speedup")
axes[2].set_title("SIMECK GPU Speedup over CPU")
axes[2].grid(True, which='both', ls='--')
axes[2].legend()

plt.tight_layout()
plt.savefig("simeck_three_columns.png")
plt.show()