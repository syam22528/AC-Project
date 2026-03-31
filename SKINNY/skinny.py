import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA (SKINNY BIT-SLICE)
# =========================
cpu_time = [
    3.64e-05,
    0.0002425,
    0.000956,
    0.0041735,
    0.0199638,
    0.0642235,
    0.174499,
    0.772229,
    1.55371
]

cpu_throughput = [
    0.45011,
    1.08101,
    1.09684,
    1.00498,
    0.840382,
    1.04493,
    0.961449,
    1.08628,
    1.07982
]

# =========================
# 🚀 GPU DATA (SKINNY BIT-SLICE)
# =========================
gpu_kernel_time = [
    0.00157901,
    8.2912e-05,
    0.00013216,
    0.000204768,
    0.000478048,
    0.00147123,
    0.00334381,
    0.0176199,
    0.0266393
]

gpu_mem_time = [
    6.3488e-05,
    6.3776e-05,
    0.000142304,
    0.000443744,
    0.00144566,
    0.00528832,
    0.0129456,
    0.0687963,
    0.141808
]

gpu_throughput = [
    0.0103761,
    3.16171,
    7.93414,
    20.4832,
    35.0953,
    45.6141,
    50.174,
    47.6088,
    62.9793
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
plt.title("SKINNY Bit-Slice CPU vs GPU Time Comparison")
plt.legend()
plt.grid()
plt.savefig("skinny_time_comparison.png")
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
plt.title("SKINNY Bit-Slice CPU vs GPU Throughput")
plt.legend()
plt.grid()
plt.savefig("skinny_throughput_comparison.png")
plt.show()

# =========================
# 🚀 SPEEDUP GRAPH
# =========================
speedup = [g/c for g, c in zip(gpu_throughput, cpu_throughput)]

plt.figure()
plt.plot(N, speedup, marker='^')
plt.xscale('log')
plt.xlabel("Number of Blocks (N)")
plt.ylabel("Speedup (GPU / CPU)")
plt.title("SKINNY GPU Speedup over CPU")
plt.grid()
plt.savefig("skinny_speedup.png")
plt.show()