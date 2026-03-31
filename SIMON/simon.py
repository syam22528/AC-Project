import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA
# =========================
cpu_time = [
    0.0001023,
    0.0016423,
    0.0065669,
    0.0262392,
    0.106454,
    0.421886,
    1.06258,
    5.20311,
    10.4873
]

cpu_throughput = [
    0.0800782,
    0.07981,
    0.079838,
    0.0799244,
    0.0788005,
    0.0795344,
    0.0789458,
    0.0806115,
    0.0799882
]

# =========================
# 🚀 GPU DATA
# =========================
gpu_kernel_time = [
    0.000134144,
    8.3968e-05,
    9.6672e-05,
    0.000155776,
    0.000261312,
    0.00078576,
    0.00175939,
    0.00834397,
    0.0129525
]

gpu_mem_time = [
    6.7584e-05,
    7.312e-05,
    0.000172736,
    0.000599264,
    0.00160733,
    0.00536154,
    0.013082,
    0.0644299,
    0.129762
]

gpu_throughput = [
    0.0610687,
    1.56098,
    5.42337,
    13.4626,
    32.1019,
    42.7032,
    47.679,
    50.2675,
    64.7645
]

# Total GPU time = kernel + memory
gpu_time = [k + m for k, m in zip(gpu_kernel_time, gpu_mem_time)]

# =========================
# 📊 TIME COMPARISON GRAPH
# =========================
plt.figure()
plt.plot(N, cpu_time, marker='o', label="CPU Time")
plt.plot(N, gpu_time , marker='s', label="GPU Time")

plt.xscale('log')
plt.yscale('log')

plt.xlabel("Number of Blocks (N)")
plt.ylabel("Time (seconds)")
plt.title("CPU vs GPU Time Comparison")
plt.legend()
plt.grid()

plt.savefig("time_comparison.png")
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
plt.title("CPU vs GPU Throughput")
plt.legend()
plt.grid()

plt.savefig("throughput_comparison.png")
plt.show()

speedup = [g/c for g, c in zip(gpu_throughput, cpu_throughput)]

plt.figure()
plt.plot(N, speedup, marker='^')
plt.xscale('log')
plt.xlabel("Number of Blocks (N)")
plt.ylabel("Speedup (GPU / CPU)")
plt.title("GPU Speedup over CPU")
plt.grid()

plt.savefig("speedup.png")
plt.show()