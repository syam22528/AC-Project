import matplotlib.pyplot as plt

# =========================
# INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# CPU DATA (TWINE BITSLICE)
# =========================
cpu_time = [
    6.57e-05,
    0.000571,
    0.0025493,
    0.009323,
    0.0367383,
    0.154083,
    0.369289,
    1.84619,
    3.70065
]

cpu_throughput = [
    0.249376,
    0.459096,
    0.411319,
    0.449888,
    0.456668,
    0.435537,
    0.454312,
    0.454374,
    0.453359
]

# =========================
# GPU DATA (TWINE BITSLICE)
# =========================
gpu_mem_time = [
    2.8672e-05,
    9.8336e-05,
    0.00015968,
    0.000476576,
    0.00144538,
    0.00526061,
    0.0129606,
    0.0640025,
    0.221733
]

gpu_kernel_time = [
    0.00142438,
    0.000145408,
    0.0001752,
    0.000260288,
    0.000747008,
    0.00257968,
    0.00625856,
    0.0310727,
    0.0620554
]

gpu_throughput = [
    0.0115025,
    1.80282,
    5.98502,
    16.1141,
    22.4592,
    26.0144,
    26.8068,
    26.9967,
    27.0359
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
axes[0].set_title("TWINE Bit-Slice CPU vs GPU Time")
axes[0].grid(True, which='both', ls='--')
axes[0].legend()

# --- Throughput Comparison ---
axes[1].plot(N, cpu_throughput, marker='o', label='CPU Throughput', color='blue')
axes[1].plot(N, gpu_throughput, marker='s', label='GPU Throughput', color='green')
axes[1].set_xscale('log')
axes[1].set_xlabel("Blocks (N)")
axes[1].set_ylabel("Throughput (GB/s)")
axes[1].set_title("TWINE Bit-Slice CPU vs GPU Throughput")
axes[1].grid(True, which='both', ls='--')
axes[1].legend()

# --- Speedup ---
axes[2].plot(N, speedup, marker='^', color='red', label='Speedup (GPU/CPU)')
axes[2].set_xscale('log')
axes[2].set_xlabel("Blocks (N)")
axes[2].set_ylabel("Speedup")
axes[2].set_title("TWINE GPU Speedup over CPU")
axes[2].grid(True, which='both', ls='--')
axes[2].legend()

plt.tight_layout()
plt.savefig("twine_three_columns.png")
plt.show()