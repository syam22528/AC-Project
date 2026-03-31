import matplotlib.pyplot as plt

# =========================
# INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# CPU DATA (WARP BITSLICE)
# =========================
cpu_time = [
    0.0001747,
    0.0024685,
    0.0095204,
    0.038339,
    0.153268,
    0.614544,
    1.5378,
    7.71469,
    15.4223
]

cpu_throughput = [
    0.0468918,
    0.0530978,
    0.05507,
    0.0547002,
    0.0547318,
    0.0546005,
    0.0545494,
    0.0543678,
    0.0543926
]

# =========================
# GPU DATA (WARP BITSLICE)
# =========================
gpu_mem_time = [
    4.7104e-05,
    0.000116864,
    0.000421632,
    0.000603776,
    0.00156957,
    0.0054472,
    0.013096,
    0.0643452,
    0.25018
]

gpu_kernel_time = [
    0.133206,
    0.00234189,
    0.00392397,
    0.0103117,
    0.0303667,
    0.10929,
    0.224808,
    1.09968,
    2.23138
]

gpu_throughput = [
    6.14987e-05,
    0.0559685,
    0.133612,
    0.203376,
    0.276243,
    0.307021,
    0.373146,
    0.38141,
    0.375938
]

# Total GPU time = kernel + memory
gpu_time = [k + m for k, m in zip(gpu_kernel_time, gpu_mem_time)]

# Compute speedup
speedup = [g / c for g, c in zip(gpu_throughput, cpu_throughput)]

# =========================
# THREE COLUMNS IN ONE FIGURE
# =========================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

# --- Time Comparison ---
axes[0].plot(N, cpu_time, marker='o', label='CPU Time', color='blue')
axes[0].plot(N, gpu_time, marker='s', label='GPU Time', color='green')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel("Blocks (N)")
axes[0].set_ylabel("Time (s)")
axes[0].set_title("WARP Bit-Slice CPU vs GPU Time")
axes[0].grid(True, which='both', ls='--')
axes[0].legend()

# --- Throughput Comparison ---
axes[1].plot(N, cpu_throughput, marker='o', label='CPU Throughput', color='blue')
axes[1].plot(N, gpu_throughput, marker='s', label='GPU Throughput', color='green')
axes[1].set_xscale('log')
axes[1].set_xlabel("Blocks (N)")
axes[1].set_ylabel("Throughput (GB/s)")
axes[1].set_title("WARP Bit-Slice CPU vs GPU Throughput")
axes[1].grid(True, which='both', ls='--')
axes[1].legend()

# --- Speedup ---
axes[2].plot(N, speedup, marker='^', color='red', label='Speedup (GPU/CPU)')
axes[2].set_xscale('log')
axes[2].set_xlabel("Blocks (N)")
axes[2].set_ylabel("Speedup")
axes[2].set_title("WARP GPU Speedup over CPU")
axes[2].grid(True, which='both', ls='--')
axes[2].legend()

plt.tight_layout()
plt.savefig("warp_three_columns.png")
plt.show()