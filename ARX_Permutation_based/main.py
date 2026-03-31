from SPECK.test_speck import *
from SPECK.benchmark_speck import *
from LEA.test_lea import *
from LEA.benchmark_lea import *
from CHAM.test_cham import *
from CHAM.benchmark_cham import *
from THREEFISH.test_threefish import *
from THREEFISH.benchmark_threefish import *
import matplotlib.pyplot as plt

def extract(results, key, skip_none=True):
    xs, ys = [], []
    for r in results:
        v = r.get(key)
        if skip_none and v is None:
            continue
        xs.append(r["size_mb"])
        ys.append(v)
    return xs, ys


if __name__ == '__main__':

    ### Testing ###
    # test_speck_naive()
    # test_speck_optimized()
    # test_speck_numba()
    # test_speck_cuda()
    # test_speck_consistency()

    # test_lea_naive()
    # test_lea_opt()
    # test_lea_numba()
    # test_lea_cuda()
    # test_lea_consistency()

    # test_cham_naive()
    # test_cham_opt()
    # test_cham_numba()
    # test_cham_cuda()
    # test_cham_consistency()


    # test_threefish_opt()
    # test_threefish_numba()
    # test_threefish_cuda()
    # test_threefish_consistency()
    

    ### Benchmarking ###
    # ascon_results = benchmark_ascon()
    speck_results = benchmark_speck()
    lea_results   = benchmark_lea()
    cham_results = benchmark_cham()
    threefish_results = benchmark_threefish()

    all_results = [speck_results, lea_results, cham_results, threefish_results]
    all_labels  = ["SPECK", "LEA", "CHAM", "THREEFISH"]
    colors = {
        "Naive":   "#e74c3c",
        "Optimized": "#e67e22",
        "Numba":   "#2980b9",
        "GPU":     "#27ae60",
        "GPU Kernel": "#1abc9c",
        "GPU Mem":  "#95a5a6",
    }
    cipher_colors = {
        "SPECK": "#2980b9",
        "LEA": "#27ae60",
        "CHAM": "#8e44ad",
        "THREEFISH": "#c0392b"
    }
    # -------------------------------------------------------------------------
    # Plot 1: Time vs Size — one subplot per cipher, all methods overlaid
    # -------------------------------------------------------------------------
    fig1, axes1 = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5))
    fig1.suptitle("Execution Time vs Input Size (per Cipher)", fontsize=14, fontweight='bold')

    if len(all_results) == 1:
        axes1 = [axes1]

    for ax, results, title in zip(axes1, all_results, all_labels):
        xs, ys = extract(results, "t_naive")
        if xs:
            ax.plot(xs, ys, marker='o', color=colors["Naive"], label="Naive", linewidth=2)

        xs, ys = extract(results, "t_opt")
        if xs:
            ax.plot(xs, ys, marker='s', color=colors["Optimized"], label="Optimized", linewidth=2)

        xs, ys = extract(results, "t_numba")
        if xs:
            ax.plot(xs, ys, marker='^', color=colors["Numba"], label="Numba", linewidth=2)

        xs, ys = extract(results, "t_gpu_total")
        if xs:
            ax.plot(xs, ys, marker='D', color=colors["GPU"], label="GPU (total)", linewidth=2)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Input Size (MB)")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("time_vs_size.png", dpi=150)
    plt.show()
    # -------------------------------------------------------------------------
    # Plot 2: Speedup (GPU over Numba) — all ciphers on one chart
    # -------------------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    fig2.suptitle("GPU Speedup over Numba vs Input Size", fontsize=14, fontweight='bold')

    for results, label in zip(all_results, all_labels):
        xs, ys = extract(results, "speedup")
        if not xs:
            continue

        ax2.plot(xs, ys, marker='o', linewidth=2, label=label, color=cipher_colors[label])

        # Safe annotation
        if ys:
            ax2.annotate(f"{ys[-1]:.1f}x", xy=(xs[-1], ys[-1]),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=8, color=cipher_colors[label])

    ax2.axhline(1, color='gray', linestyle='--', linewidth=1, label="1x (no speedup)")
    ax2.set_xlabel("Input Size (MB)")
    ax2.set_ylabel("Speedup (GPU total / Numba)")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("speedup.png", dpi=150)
    plt.show()
    # -------------------------------------------------------------------------
    # Plot 3: Throughput (MB/s) — GPU vs Numba per cipher
    # -------------------------------------------------------------------------
    fig3, axes3 = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5))
    fig3.suptitle("Throughput (MB/s) vs Input Size", fontsize=14, fontweight='bold')

    if len(all_results) == 1:
        axes3 = [axes3]

    for ax, results, title in zip(axes3, all_results, all_labels):
        xs, ys = extract(results, "gpu_throughput")
        if xs:
            ax.plot(xs, ys, marker='D', color=colors["GPU"], label="GPU", linewidth=2)

        xs2, ys2 = extract(results, "numba_throughput")
        if xs2:
            ax.plot(xs2, ys2, marker='^', color=colors["Numba"], label="Numba",
                    linewidth=2, linestyle='--')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Input Size (MB)")
        ax.set_ylabel("Throughput (MB/s)")
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("throughput.png", dpi=150)
    plt.show()
    # -------------------------------------------------------------------------
    # Plot 4: GPU time breakdown — kernel vs memory transfer per cipher
    # -------------------------------------------------------------------------
    fig4, axes4 = plt.subplots(1, len(all_results), figsize=(5 * len(all_results), 5))
    fig4.suptitle("GPU Time Breakdown: Kernel vs Memory Transfer", fontsize=14, fontweight='bold')

    if len(all_results) == 1:
        axes4 = [axes4]

    for ax, results, title in zip(axes4, all_results, all_labels):
        xs_k, ys_k = extract(results, "t_gpu_kernel")
        xs_m, ys_m = extract(results, "t_gpu_mem")

        if xs_k:
            ax.plot(xs_k, ys_k, marker='o', color=colors["GPU Kernel"],
                    label="Kernel compute", linewidth=2)

        if xs_m:
            ax.plot(xs_m, ys_m, marker='s', color=colors["GPU Mem"],
                    label="Memory transfer", linewidth=2, linestyle='--')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Input Size (MB)")
        ax.set_ylabel("Time (s)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, which='both', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("gpu_breakdown.png", dpi=150)
    plt.show()
    # -------------------------------------------------------------------------
    # Plot 5: Side-by-side bar chart — time at 100 MB for all methods & ciphers
    # -------------------------------------------------------------------------
    import numpy as np

    fig5, ax5 = plt.subplots(figsize=(10, 6))
    fig5.suptitle("Execution Time at 100 MB — All Methods & Ciphers", fontsize=14, fontweight='bold')

    bar_methods = ["Numba", "GPU (total)", "GPU (kernel)", "GPU (mem)"]
    bar_keys    = ["t_numba", "t_gpu_total", "t_gpu_kernel", "t_gpu_mem"]
    bar_colors  = [colors["Numba"], colors["GPU"], colors["GPU Kernel"], colors["GPU Mem"]]

    x = np.arange(len(all_labels))
    n_bars = len(bar_methods)
    width  = 0.18

    for i, (method, key, color) in enumerate(zip(bar_methods, bar_keys, bar_colors)):
        vals = []
        for results in all_results:
            # last row = 100 MB
            vals.append(results[-1][key])
        offset = (i - n_bars / 2 + 0.5) * width
        bars = ax5.bar(x + offset, vals, width, label=method, color=color)
        for bar, v in zip(bars, vals):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f"{v:.3f}s", ha='center', va='bottom', fontsize=7, rotation=45)

    ax5.set_xticks(x)
    ax5.set_xticklabels(all_labels, fontsize=12)
    ax5.set_ylabel("Time (s)")
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("bar_100mb.png", dpi=150)
    plt.show()