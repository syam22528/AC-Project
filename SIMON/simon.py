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
    0.0015937,
    0.0063998,
    0.0258844,
    0.102515,
    0.410847,
    1.03419
]

cpu_throughput = [
    0.0820842,
    0.0822438,
    0.0819226,
    0.0810199,
    0.0818281,
    0.0816715,
    0.0811126
]

# =========================
# 🚀 GPU DATA (SIMON)
# =========================
gpu_kernel_time = [
    0.0001536,
    8.704e-05,
    9.344e-05,
    0.000324032,
    0.000369088,
    0.000865696,
    0.00186858
]

# For SIMON GPU, we don't have separate memory times, so total GPU time = kernel time
gpu_time = gpu_kernel_time

gpu_throughput = [
    0.0533333,
    1.50588,
    5.61096,
    6.47205,
    22.7279,
    38.7601,
    44.8931
]

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
plt.savefig("simon_three_columns.png")
plt.show()