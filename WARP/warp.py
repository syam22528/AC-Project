import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA (WARP BIT-SLICE + OpenMP)
# =========================
cpu_time = [
    0.0001178,
    0.002683,
    0.0079394,
    0.0307504,
    0.123118,
    0.491362,
    1.22955,
    6.16436,
    12.3847
]

cpu_throughput = [
    0.139083,
    0.0977056,
    0.132072,
    0.136398,
    0.13627,
    0.136577,
    0.13645,
    0.136082,
    0.135467
]

# =========================
# 🚀 GPU DATA (WARP BIT-SLICE OPTIMIZED)
# =========================
gpu_kernel_time = [
    0.0027351,
    0.00110285,
    0.00111411,
    0.0032103,
    0.0120146,
    0.0407316,
    0.110266,
    0.432332,
    0.837368
]

gpu_mem_time = [
    4.4032e-05,
    0.0002,
    0.000216928,
    0.000665056,
    0.00153789,
    0.00551933,
    0.0145383,
    0.0709997,
    0.254329
]

gpu_time = [k + m for k, m in zip(gpu_kernel_time, gpu_mem_time)]

gpu_throughput = [
    0.00599027,
    0.237697,
    0.941176,
    1.30651,
    1.3964,
    1.64759,
    1.52152,
    1.94032,
    2.00357
]

# =========================
# 📊 TIME COMPARISON GRAPH
# =========================
plt.figure()
plt.plot(N, cpu_time, marker='o', label="CPU Time")
plt.plot(N, gpu_time, marker='s', label="GPU Time")

plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Blocks (N)")
plt.ylabel("Time (seconds)")
plt.title("WARP Bit-Slice CPU vs GPU Time Comparison")
plt.legend()
plt.grid()
plt.savefig("warp_time_comparison.png")
plt.show()

# =========================
# 📊 THROUGHPUT GRAPH
# =========================
plt.figure()
plt.plot(N, cpu_throughput, marker='o', label="CPU Throughput")
plt.plot(N, gpu_throughput, marker='s', label="GPU Throughput")

plt.xscale('log')
plt.xlabel("Number of Blocks (N)")
plt.ylabel("Throughput (GB/s)")
plt.title("WARP Bit-Slice CPU vs GPU Throughput")
plt.legend()
plt.grid()
plt.savefig("warp_throughput_comparison.png")
plt.show()

# =========================
# 🚀 SPEEDUP GRAPH
# =========================
speedup = [g / c for g, c in zip(gpu_throughput, cpu_throughput)]

plt.figure()
plt.plot(N, speedup, marker='^')
plt.xscale('log')
plt.xlabel("Number of Blocks (N)")
plt.ylabel("Speedup (GPU / CPU)")
plt.title("WARP GPU Speedup over CPU")
plt.grid()
plt.savefig("warp_speedup.png")
plt.show()