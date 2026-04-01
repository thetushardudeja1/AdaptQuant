# stress_test.py — sustained 30 minute inference stress test
# Logs thermal behaviour, stress scores, and switch events over time
# Produces the time-series data for the paper's key figure

import os
import sys
import time
import csv
import json
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/AdaptQuant"))

import orchestrator.hot_swap as hs
from orchestrator.policy import decide, get_stress, reset
from scripts.monitor import get_metrics
from scripts.logger import log_switch

# ── Config ────────────────────────────────────────────────────────────────────
DURATION_SEC   = 1800    # 30 minutes
START_DOMAIN   = "general"
START_QUANT    = "Q5"
MAX_TOKENS     = 150

RESULTS_DIR    = os.path.expanduser("~/AdaptQuant/results")
STRESS_LOG     = os.path.join(RESULTS_DIR, "stress_test.csv")
STRESS_SUMMARY = os.path.join(RESULTS_DIR, "stress_summary.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Rotating prompts to keep inference varied ─────────────────────────────────
PROMPTS = [
    "Explain how neural networks learn through backpropagation in detail.",
    "Write a Python function to implement quicksort with explanation.",
    "Describe the causes and effects of climate change comprehensively.",
    "Explain the difference between TCP and UDP protocols in networking.",
    "Write a detailed explanation of how transformers work in NLP.",
    "Describe how the human immune system responds to viral infections.",
    "Explain recursion with multiple examples in computer science.",
    "Write a Python class implementing a binary search tree with insert and search.",
    "Describe the history and impact of the industrial revolution.",
    "Explain gradient descent and its variants in machine learning.",
    "Write a function to find all prime numbers up to N using a sieve.",
    "Describe how database indexing works and why it improves performance.",
    "Explain the concept of entropy in thermodynamics and information theory.",
    "Write a detailed explanation of how operating system scheduling works.",
    "Describe the architecture of a convolutional neural network.",
]

# ── CSV logger ────────────────────────────────────────────────────────────────
def append_csv(row):
    write_header = not os.path.exists(STRESS_LOG)
    with open(STRESS_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ── Main stress test ──────────────────────────────────────────────────────────
def run():
    current_quant  = START_QUANT
    current_domain = START_DOMAIN
    prompt_idx     = 0
    switch_count   = 0
    inference_count = 0

    models = {
        ("general", "Q5"): "~/AdaptQuant/models/general-Q5.gguf",
        ("general", "Q3"): "~/AdaptQuant/models/general-Q3.gguf",
    }

    print("=" * 60)
    print("  AdaptQuant Stress Test — 30 minutes")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Initialize
    hs.initialize(models[(current_domain, current_quant)])
    start_time = time.time()

    try:
        while time.time() - start_time < DURATION_SEC:
            elapsed   = round(time.time() - start_time)
            remaining = DURATION_SEC - elapsed

            # Get system metrics
            metrics = get_metrics()
            stress  = get_stress(metrics["cpu_temp_c"])

            # Pick next prompt
            prompt = PROMPTS[prompt_idx % len(PROMPTS)]
            prompt_idx += 1

            # Run inference
            result = hs.query(prompt, max_tokens=MAX_TOKENS)
            if not result:
                print(f"  [{elapsed}s] Query failed, retrying...")
                time.sleep(2)
                continue

            inference_count += 1

            # Log everything
            row = {
                "elapsed_sec":  elapsed,
                "timestamp":    datetime.now().isoformat(),
                "model":        f"{current_domain}-{current_quant}",
                "quant":        current_quant,
                "gen_tps":      result["gen_tps"],
                "ttft_ms":      result["ttft_ms"],
                "cpu_temp_c":   metrics["cpu_temp_c"],
                "ram_used_mb":  metrics["ram_used_mb"],
                "ram_percent":  metrics["ram_percent"],
                "stress_score": stress,
                "switch_count": switch_count,
            }
            append_csv(row)

            print(f"  [{elapsed:>4}s | {remaining:>4}s left] "
                  f"{current_domain}-{current_quant} | "
                  f"{result['gen_tps']} tok/s | "
                  f"{metrics['cpu_temp_c']}°C | "
                  f"stress: {stress} | "
                  f"switches: {switch_count}")

            # Check switching policy
            new_quant, reason = decide(
                cpu_temp      = metrics["cpu_temp_c"],
                ram_percent   = metrics["ram_percent"],
                gen_tps       = result["gen_tps"],
                current_quant = current_quant,
            )

            if new_quant:
                print(f"\n  *** SWITCH: {current_quant} → {new_quant} | {reason} ***\n")
                new_path   = models[(current_domain, new_quant)]
                latency_ms = hs.swap(new_path)
                switch_count += 1

                log_switch(
                    from_model        = f"{current_domain}-{current_quant}",
                    to_model          = f"{current_domain}-{new_quant}",
                    reason            = reason,
                    switch_latency_ms = latency_ms,
                    cpu_temp          = metrics["cpu_temp_c"],
                    ram_mb            = metrics["ram_used_mb"],
                )

                current_quant = new_quant
                print(f"  Now running {current_domain}-{current_quant}\n")

    except KeyboardInterrupt:
        print("\n  Stress test interrupted by user.")

    finally:
        hs.shutdown()
        elapsed_total = round(time.time() - start_time)

        summary = {
            "duration_sec":    elapsed_total,
            "total_inferences": inference_count,
            "total_switches":  switch_count,
            "started":         datetime.now().isoformat(),
            "start_quant":     START_QUANT,
            "end_quant":       current_quant,
        }

        with open(STRESS_SUMMARY, "w") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("  Stress test complete!")
        print(f"  Duration     : {elapsed_total}s")
        print(f"  Inferences   : {inference_count}")
        print(f"  Switches     : {switch_count}")
        print(f"  Results      : {STRESS_LOG}")
        print("=" * 60)

if __name__ == "__main__":
    run()