import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA (TWINE BIT-SLICE)
# =========================
cpu_time = [
    9.03e-05,
    0.0010759,
    0.0038949,
    0.0156107,
    0.0632905,
    0.249998,
    0.625969,
    3.15895,
    6.30134
]

cpu_throughput = [
    0.18144,
    0.243651,
    0.269218,
    0.268681,
    0.265083,
    0.268438,
    0.26802,
    0.26555,
    0.266248
]

# =========================
# 🚀 GPU DATA (TWINE BIT-SLICE)
# =========================
gpu_kernel_time = [
    0.0014889,
    0.000133248,
    0.000111392,
    0.000212736,
    0.000576448,
    0.00181709,
    0.00423453,
    0.0207108,
    0.0316487
]

gpu_mem_time = [
    3.1808e-05,
    7.9744e-05,
    0.000156288,
    0.000452992,
    0.00182125,
    0.00533264,
    0.0139297,
    0.0639872,
    0.128003
]

gpu_time = [k + m for k, m in zip(gpu_kernel_time, gpu_mem_time)]

gpu_throughput = [
    0.0110041,
    1.96734,
    9.41339,
    19.716,
    29.1045,
    36.9321,
    39.62,
    40.5035,
    53.0107
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
plt.title("TWINE Bit-Slice CPU vs GPU Time Comparison")
plt.legend()
plt.grid()
plt.savefig("twine_time_comparison.png")
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
plt.title("TWINE Bit-Slice CPU vs GPU Throughput")
plt.legend()
plt.grid()
plt.savefig("twine_throughput_comparison.png")
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
plt.title("TWINE GPU Speedup over CPU")
plt.grid()
plt.savefig("twine_speedup.png")
plt.show()