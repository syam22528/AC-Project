import matplotlib.pyplot as plt

# =========================
# 📊 INPUT SIZES
# =========================
N = [1024, 16384, 65536, 262144, 1048576, 4194304, 10485760, 52428800, 104857600]

# =========================
# 🖥️ CPU DATA (SIMECK)
# =========================
cpu_time = [
    9.51e-05,
    0.0015192,
    0.0061018,
    0.0244483,
    0.0977708,
    0.391393,
    0.981856,
    4.9163,
    9.81888
]

cpu_throughput = [
    0.0861409,
    0.086277,
    0.0859235,
    0.0857791,
    0.0857987,
    0.0857308,
    0.0854363,
    0.0853143,
    0.0854335
]

# =========================
# 🚀 GPU DATA (SIMECK)
# =========================
gpu_kernel_time = [
    0.000130048,
    8.3968e-05,
    5.3248e-05,
    0.00011376,
    0.000225024,
    0.000657088,
    0.00152554,
    0.00717846,
    0.0112031
]

gpu_mem_time = [
    0.000121856,
    0.000131104,
    0.000165824,
    0.000555648,
    0.00156419,
    0.00550502,
    0.0130573,
    0.0644676,
    0.128384
]

gpu_throughput = [
    0.0629921,
    1.56098,
    9.84615,
    18.4349,
    37.2787,
    51.0654,
    54.9879,
    58.429,
    74.8778
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
plt.title("SIMECK: CPU vs GPU Time Comparison")
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
plt.title("SIMECK: CPU vs GPU Throughput")
plt.legend()
plt.grid()

plt.savefig("throughput_comparison.png")
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
plt.title("SIMECK GPU Speedup over CPU")
plt.grid()

plt.savefig("speedup.png")
plt.show()