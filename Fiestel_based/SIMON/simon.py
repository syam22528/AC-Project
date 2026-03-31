import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760]

# =========================
# 🖥️ CPU DATA (SIMON)
# =========================
cpu_time = [
    9.98e-05,
    0.001602,
    0.0065641,
    0.0256668,
    0.103192,
    0.416403,
    1.03092
]

cpu_throughput = [
    0.0820842,
    0.0818177,
    0.079872,
    0.0817068,
    0.0812916,
    0.0805816,
    0.0813705
]

# =========================
# 🚀 GPU DATA (SIMON)
# =========================
gpu_kernel_time = [
    0.000130048,
    3.3792e-05,
    8.5984e-05,
    0.00153667,
    0.000355264,
    0.000819936,
    0.00183376
]

gpu_mem_time = [
    4.9152e-05,
    7.8592e-05,
    0.000181728,
    0.00070048,
    0.00162384,
    0.00700266,
    0.0137714
]

gpu_throughput = [
    0.0629921,
    3.87879,
    6.09751,
    1.36474,
    23.6123,
    40.9232,
    45.7454
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
axes[0].set_title("SIMON CPU vs GPU Time")
axes[0].grid(True, which='both', ls='--')
axes[0].legend()

# --- Throughput Comparison ---
axes[1].plot(N, cpu_throughput, marker='o', label='CPU Throughput', color='blue')
axes[1].plot(N, gpu_throughput, marker='s', label='GPU Throughput', color='green')
axes[1].set_xscale('log')
axes[1].set_xlabel("Blocks (N)")
axes[1].set_ylabel("Throughput (GB/s)")
axes[1].set_title("SIMON CPU vs GPU Throughput")
axes[1].grid(True, which='both', ls='--')
axes[1].legend()

# --- Speedup ---
axes[2].plot(N, speedup, marker='^', color='red', label='Speedup (GPU/CPU)')
axes[2].set_xscale('log')
axes[2].set_xlabel("Blocks (N)")
axes[2].set_ylabel("Speedup")
axes[2].set_title("SIMON GPU Speedup over CPU")
axes[2].grid(True, which='both', ls='--')
axes[2].legend()

plt.tight_layout()
plt.savefig("simon_three_columns_updated.png")
plt.show()