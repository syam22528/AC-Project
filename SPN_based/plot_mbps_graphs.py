from __future__ import annotations

"""Benchmark visualisation: 5 focused figures for ECB and CTR modes.

Reads CSV results from SNP_Ciphers/results/ and writes PNG plots to
SNP_Ciphers/results/plots/.  The five output figures are:

  throughput_ecb.png    — 1×4 grid: ECB throughput (MB/s) per cipher.
  throughput_ctr.png    — 1×4 grid: CTR throughput (MB/s) per cipher.
  speedup_ecb.png       — 1×4 grid: ECB GPU speedup over CPU per cipher.
  speedup_ctr.png       — 1×4 grid: CTR GPU speedup over CPU per cipher.
  cipher_comparison.png — 2×2 grid: best-variant cross-cipher summary.

Note: AES has one GPU variant (shared-memory S-box); GIFT, SKINNY, and
PRESENT each have two variants (table and bitsliced).
"""

import csv
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Apply consistent style across all figures before any plot is created.
matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linewidth":    0.7,
    "lines.linewidth":   2.2,
    "lines.markersize":  7,
})

ROOT    = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

# Per-cipher brand colours for consistent visual identity across all plots.
CIPHER_COLOR = {
    "AES":     "#2f6eb5",
    "GIFT":    "#d4612a",
    "SKINNY":  "#2e8b57",
    "PRESENT": "#b22222",
}

# Line styles for variant 0 (solid + circle) and variant 1 (dashed + square).
VARIANT_STYLES = [
    {"ls": "-",  "marker": "o"},   # first variant (table / shared)
    {"ls": "--", "marker": "s"},   # second variant (bitsliced)
]

# CPU line style: dotted grey, low opacity so GPU lines remain prominent.
CPU_KW = dict(ls=":", marker="x", lw=1.6, color="#555555", alpha=0.75, zorder=2)

CIPHERS = ("AES", "GIFT", "SKINNY", "PRESENT")

# Maps each cipher to its CSV filename, the column that distinguishes variants,
# and any additional row filters (empty dict = accept all rows).
CIPHER_CFG = {
    "AES":     ("aes_benchmark.csv",     "key_mode", {}),
    "GIFT":    ("gift_benchmark.csv",    "variant",  {}),
    "SKINNY":  ("skinny_benchmark.csv",  "variant",  {}),
    "PRESENT": ("present_benchmark.csv", "variant",  {}),
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _f(v: str) -> float: return float(v.strip())
def _i(v: str) -> int:   return int(v.strip())


def load(cipher: str) -> list[dict]:
    """Load all matching rows from the cipher's CSV file.

    Skips header-repeat rows (where 'mode' == 'mode') and rows that do not
    satisfy any extra filter conditions defined in CIPHER_CFG.
    """
    fname, _, filters = CIPHER_CFG[cipher]
    path = RESULTS / fname
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="ascii") as f:
        for r in csv.DictReader(f):
            if not r or r.get("mode", "").strip().lower() == "mode":
                continue
            if all(r.get(k, "").strip() == v for k, v in filters.items()):
                rows.append(r)
    return rows


def get_variants(cipher: str) -> list[str]:
    """Return the sorted list of unique variant values found in the cipher's CSV."""
    _, vcol, _ = CIPHER_CFG[cipher]
    return sorted({r[vcol] for r in load(cipher)})


def series(rows: list[dict], mode: str, vcol: str, variant: str):
    """Extract sorted (blocks, gpu_mbps, cpu_mbps, speedup) series for one variant/mode."""
    sel = sorted(
        [r for r in rows if r["mode"] == mode and r[vcol] == variant],
        key=lambda r: _i(r["blocks"]),
    )
    if not sel:
        return [], [], [], []
    return (
        [_i(r["blocks"])         for r in sel],
        [_f(r["gpu_total_mbps"]) for r in sel],
        [_f(r["cpu_mbps"])       for r in sel],
        [_f(r["speedup_total"])  for r in sel],
    )


def best_series(cipher: str, mode: str):
    """Return the best-variant series for `cipher` in `mode`.

    For each block count, selects the row with the highest gpu_total_mbps
    across all variants, giving the 'best achievable' GPU throughput line.
    """
    rows = load(cipher)
    _, vcol, _ = CIPHER_CFG[cipher]
    best: dict[int, dict] = {}
    for r in rows:
        if r["mode"] != mode:
            continue
        b = _i(r["blocks"])
        if b not in best or _f(r["gpu_total_mbps"]) > _f(best[b]["gpu_total_mbps"]):
            best[b] = r
    xs = sorted(best)
    return (
        xs,
        [_f(best[x]["gpu_total_mbps"]) for x in xs],
        [_f(best[x]["cpu_mbps"])       for x in xs],
        [_f(best[x]["speedup_total"])  for x in xs],
    )


def fmt_blocks(v, _):
    """Format a block count as a short human-readable label (e.g. 1M, 64k)."""
    v = int(v)
    if v >= 1_000_000: return f"{v // 1_000_000}M"
    if v >= 1_000:     return f"{v // 1_000}k"
    return str(v)


def _style_ax(ax, ylabel="", xlabel=True, xs=None):
    """Apply common axis formatting: log2 x-scale, tick labels, grid, y-floor."""
    ax.set_xscale("log", base=2)
    if xs is not None and len(xs) > 0:
        ax.set_xticks(xs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_blocks))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_ylim(bottom=0)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Blocks")
    ax.tick_params(axis='x', rotation=45)


# ---------------------------------------------------------------------------
# Panel drawing functions
# ---------------------------------------------------------------------------

def _draw_throughput_panel(ax, cipher: str, mode: str):
    """Plot GPU throughput lines (one per variant) and a single CPU reference line."""
    rows = load(cipher)
    _, vcol, _ = CIPHER_CFG[cipher]
    variants = get_variants(cipher)
    color = CIPHER_COLOR[cipher]

    cpu_added = False
    all_xs = set()
    for vi, variant in enumerate(variants):
        xs, gpu_ys, cpu_ys, _ = series(rows, mode, vcol, variant)
        if not xs:
            continue
        all_xs.update(xs)
        sty = VARIANT_STYLES[vi % len(VARIANT_STYLES)]
        ax.plot(xs, gpu_ys, color=color, label=f"GPU – {variant}", **sty)
        # Draw the CPU line only once (it is the same for all variants).
        if not cpu_added:
            ax.plot(xs, cpu_ys, label="CPU", **CPU_KW)
            cpu_added = True

    ax.set_title(cipher, fontweight="bold", color=color, pad=10)
    _style_ax(ax, ylabel="MB/s", xs=sorted(all_xs))
    ax.legend(framealpha=0.85, loc="upper left")


def _draw_speedup_panel(ax, cipher: str, mode: str):
    """Plot GPU/CPU speedup lines (one per variant) with a 1× reference line."""
    rows = load(cipher)
    _, vcol, _ = CIPHER_CFG[cipher]
    variants = get_variants(cipher)
    color = CIPHER_COLOR[cipher]

    all_xs = set()
    for vi, variant in enumerate(variants):
        xs, _, _, sp = series(rows, mode, vcol, variant)
        if not xs:
            continue
        all_xs.update(xs)
        sty = VARIANT_STYLES[vi % len(VARIANT_STYLES)]
        ax.plot(xs, sp, color=color, label=variant, **sty)

    # Horizontal line at speedup = 1 (GPU equals CPU performance).
    ax.axhline(1.0, color="#aaaaaa", lw=1.1, ls="--")
    ax.set_title(cipher, fontweight="bold", color=color, pad=10)
    _style_ax(ax, ylabel="Speedup (×)", xs=sorted(all_xs))
    ax.legend(title="GPU variant", framealpha=0.85, loc="upper left")


# ---------------------------------------------------------------------------
# Figure constructors (1×4 rows and the 2×2 summary)
# ---------------------------------------------------------------------------

def _make_row(draw_fn, mode: str, title: str, fname: str) -> str:
    """Create a 1×4 figure calling `draw_fn` for each cipher, save to disk."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold")

    for ax, cipher in zip(axes, CIPHERS):
        draw_fn(ax, cipher, mode)

    out = PLOTS / fname
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def fig_throughput_ecb() -> str:
    """Generate and save Figure 1: ECB throughput (MB/s), 1×4 cipher grid."""
    return _make_row(
        _draw_throughput_panel, "ecb",
        "GPU vs CPU Throughput — ECB Mode (MB/s)",
        "throughput_ecb.png",
    )


def fig_throughput_ctr() -> str:
    """Generate and save Figure 2: CTR throughput (MB/s), 1×4 cipher grid."""
    return _make_row(
        _draw_throughput_panel, "ctr",
        "GPU vs CPU Throughput — CTR Mode (MB/s)",
        "throughput_ctr.png",
    )


def fig_speedup_ecb() -> str:
    """Generate and save Figure 3: ECB GPU speedup over CPU, 1×4 cipher grid."""
    return _make_row(
        _draw_speedup_panel, "ecb",
        "GPU Speedup over CPU — ECB Mode (×)",
        "speedup_ecb.png",
    )


def fig_speedup_ctr() -> str:
    """Generate and save Figure 4: CTR GPU speedup over CPU, 1×4 cipher grid."""
    return _make_row(
        _draw_speedup_panel, "ctr",
        "GPU Speedup over CPU — CTR Mode (×)",
        "speedup_ctr.png",
    )


def fig_comparison() -> str:
    """Generate and save Figure 5: 2×2 best-variant cross-cipher summary.

    Each of the four panels shows all four ciphers plotted together using
    their best-performing GPU variant, for quick cross-cipher comparison.
    Panels: top row = throughput (ECB, CTR); bottom row = speedup (ECB, CTR).
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
    fig.suptitle("Best-Variant Summary: All Ciphers", fontsize=15, fontweight="bold")

    panels = [
        (axes[0][0], "ecb", False, "ECB — GPU Throughput (MB/s)",    "MB/s"),
        (axes[0][1], "ctr", False, "CTR — GPU Throughput (MB/s)",    "MB/s"),
        (axes[1][0], "ecb", True,  "ECB — GPU Speedup over CPU (×)", "Speedup (×)"),
        (axes[1][1], "ctr", True,  "CTR — GPU Speedup over CPU (×)", "Speedup (×)"),
    ]

    for ax, mode, is_speedup, title, ylabel in panels:
        all_xs = set()
        for cipher in CIPHERS:
            xs, gpu_ys, cpu_ys, sp_ys = best_series(cipher, mode)
            if not xs:
                continue
            all_xs.update(xs)
            color = CIPHER_COLOR[cipher]
            ys = sp_ys if is_speedup else gpu_ys
            ax.plot(xs, ys, color=color, lw=2.4, marker="o", ms=7, label=cipher)
            if not is_speedup:
                # Show faint CPU reference lines for throughput panels.
                ax.plot(xs, cpu_ys, color=color, lw=1.2, ls="--",
                        marker="x", ms=5, alpha=0.4)

        if is_speedup:
            ax.axhline(1.0, color="#aaaaaa", lw=1.1, ls="--")

        ax.set_title(title, fontweight="bold", pad=10)
        _style_ax(ax, ylabel=ylabel, xlabel=True, xs=sorted(all_xs))
        ax.legend(framealpha=0.85, loc="upper left", ncol=2)

    out = PLOTS / "cipher_comparison.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Generate all five benchmark figures and print their output paths."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        generated = [
            fig_throughput_ecb(),
            fig_throughput_ctr(),
            fig_speedup_ecb(),
            fig_speedup_ctr(),
            fig_comparison(),
        ]
    print(f"Generated {len(generated)} plots:")
    for p in generated:
        print(" ", p)


if __name__ == "__main__":
    main()
