# generate_graphs.py — generates all paper figures from benchmark CSVs

import os
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for Pi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = os.path.expanduser("~/AdaptQuant/results")
GRAPHS_DIR  = os.path.join(RESULTS_DIR, "graphs")
os.makedirs(GRAPHS_DIR, exist_ok=True)

def save(name):
    path = os.path.join(GRAPHS_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ── Figure 1 — Throughput comparison (tok/s) ─────────────────────────────────
def fig_throughput():
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json")) as f:
        summary = json.load(f)

    models  = [s["model"] for s in summary]
    tps     = [s["avg_gen_tps"] for s in summary]
    colors  = ["#4C72B0" if s["quant"] == "Q5" else "#DD8452" for s in summary]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(models, tps, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f"{val}", ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Tokens per second (tok/s)", fontsize=12)
    ax.set_title("Figure 1 — Inference throughput by model and quantization level",
                 fontsize=13, pad=15)
    ax.set_ylim(0, max(tps) * 1.15)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    q5_patch = mpatches.Patch(color="#4C72B0", label="Q5 (higher quality)")
    q3_patch = mpatches.Patch(color="#DD8452", label="Q3 (higher speed)")
    ax.legend(handles=[q5_patch, q3_patch], fontsize=10)

    plt.tight_layout()
    save("fig1_throughput.png")

# ── Figure 2 — Perplexity heatmap ────────────────────────────────────────────
def fig_perplexity():
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json")) as f:
        summary = json.load(f)

    models      = [s["model"] for s in summary]
    domains     = ["general", "code", "chat"]
    ppl_matrix  = np.array([
        [s["ppl_general"], s["ppl_code"], s["ppl_chat"]]
        for s in summary
    ])

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(ppl_matrix, cmap="RdYlGn_r", aspect="auto")

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(["General corpus", "Code corpus", "Chat corpus"],
                       fontsize=11)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=10)

    for i in range(len(models)):
        for j in range(len(domains)):
            val = ppl_matrix[i, j]
            col = "white" if val > np.median(ppl_matrix) else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=10, color=col, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Perplexity (lower = better)")
    ax.set_title("Figure 2 — Cross-domain perplexity matrix\n"
                 "(diagonal = domain-matched models)",
                 fontsize=13, pad=15)
    plt.tight_layout()
    save("fig2_perplexity_heatmap.png")

# ── Figure 3 — Stress test time series ───────────────────────────────────────
def fig_stress_test():
    df = pd.read_csv(os.path.join(RESULTS_DIR, "stress_test.csv"))

    # Keep only the full 30 min run (elapsed resets each test run)
    # Use the longest continuous run
    runs = []
    current_run = []
    prev = -1
    for _, row in df.iterrows():
        if row["elapsed_sec"] < prev:
            runs.append(current_run)
            current_run = []
        current_run.append(row)
        prev = row["elapsed_sec"]
    runs.append(current_run)
    longest = max(runs, key=len)
    df = pd.DataFrame(longest).reset_index(drop=True)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Switch moment
    switch_rows = df[df["switch_count"].diff() > 0]

    # Panel 1 — Temperature
    ax1.plot(df["elapsed_sec"], df["cpu_temp_c"],
             color="#E05C5C", linewidth=1.5, label="CPU temp")
    ax1.axhline(72, color="red", linestyle="--", alpha=0.5, linewidth=1,
                label="Hard ceiling (72°C)")
    ax1.axhline(60, color="orange", linestyle="--", alpha=0.5, linewidth=1,
                label="PID target (60°C)")
    for _, row in switch_rows.iterrows():
        ax1.axvline(row["elapsed_sec"], color="blue", alpha=0.4, linewidth=1.5)
    ax1.set_ylabel("CPU temp (°C)", fontsize=11)
    ax1.set_ylim(40, 85)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.grid(alpha=0.3)
    ax1.set_title("Figure 3 — AdaptQuant stress test: 30 minutes continuous inference",
                  fontsize=13, pad=10)

    # Panel 2 — Throughput
    q5 = df[df["quant"] == "Q5"]
    q3 = df[df["quant"] == "Q3"]
    ax2.scatter(q5["elapsed_sec"], q5["gen_tps"],
                color="#4C72B0", s=15, label="Q5", alpha=0.8)
    ax2.scatter(q3["elapsed_sec"], q3["gen_tps"],
                color="#DD8452", s=15, label="Q3", alpha=0.8)
    for _, row in switch_rows.iterrows():
        ax2.axvline(row["elapsed_sec"], color="blue", alpha=0.4, linewidth=1.5)
    ax2.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax2.set_ylim(10, 25)
    ax2.legend(fontsize=9, loc="upper right")
    ax2.grid(alpha=0.3)

    # Panel 3 — Stress score
    ax3.plot(df["elapsed_sec"], df["stress_score"],
             color="#6B4EAE", linewidth=1.2, alpha=0.8, label="PID stress score")
    ax3.axhline(60, color="red", linestyle="--", alpha=0.5, linewidth=1,
                label="Switch threshold (60)")
    ax3.axhline(30, color="green", linestyle="--", alpha=0.5, linewidth=1,
                label="Recovery threshold (30)")
    for _, row in switch_rows.iterrows():
        ax3.axvline(row["elapsed_sec"], color="blue", alpha=0.4, linewidth=1.5,
                    label="Switch event" if _ == switch_rows.index[0] else "")
    ax3.set_ylabel("PID stress score", fontsize=11)
    ax3.set_xlabel("Elapsed time (seconds)", fontsize=11)
    ax3.set_ylim(-5, 110)
    ax3.legend(fontsize=9, loc="upper right")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    save("fig3_stress_test.png")

# ── Figure 4 — TTFT comparison ───────────────────────────────────────────────
def fig_ttft():
    with open(os.path.join(RESULTS_DIR, "benchmark_summary.json")) as f:
        summary = json.load(f)

    models  = [s["model"] for s in summary]
    avg_ttft = [s["avg_ttft_ms"] for s in summary]
    p90_ttft = [s["p90_ttft_ms"] for s in summary]
    colors   = ["#4C72B0" if s["quant"] == "Q5" else "#DD8452" for s in summary]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, avg_ttft, width, label="Avg TTFT",
                   color=colors, alpha=0.9, edgecolor="white")
    bars2 = ax.bar(x + width/2, p90_ttft, width, label="P90 TTFT",
                   color=colors, alpha=0.5, edgecolor="white", hatch="//")

    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Figure 4 — Time to first token: average and P90 latency",
                 fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save("fig4_ttft.png")

# ── Run all ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    fig_throughput()
    fig_perplexity()
    fig_stress_test()
    fig_ttft()
    print(f"\nAll figures saved to {GRAPHS_DIR}")