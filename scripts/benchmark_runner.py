# benchmark_runner.py — full automated benchmark suite for AdaptQuant
# Runs unattended overnight. Logs all industry-grade metrics to CSV.

import os
import sys
import time
import csv
import json
import subprocess
import requests
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/AdaptQuant"))
from scripts.monitor import get_metrics

# ── Config ────────────────────────────────────────────────────────────────────
LLAMA_SERVER    = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
LLAMA_PERP      = os.path.expanduser("~/llama.cpp/build/bin/llama-perplexity")
HOST            = "127.0.0.1"
PORT            = 8080
THREADS         = 4
CTX_SIZE        = 512
COOLDOWN_SEC    = 60
WARMUP_PROMPTS  = 2
BENCH_PROMPTS   = 10
MAX_TOKENS      = 100

RESULTS_DIR     = os.path.expanduser("~/AdaptQuant/results")
INFERENCE_CSV   = os.path.join(RESULTS_DIR, "benchmark_inference.csv")
PERPLEXITY_CSV  = os.path.join(RESULTS_DIR, "benchmark_perplexity.csv")
SUMMARY_JSON    = os.path.join(RESULTS_DIR, "benchmark_summary.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Models ────────────────────────────────────────────────────────────────────
MODELS = [
    {"name": "general-Q5", "domain": "general", "quant": "Q5",
     "path": "~/AdaptQuant/models/general-Q5.gguf"},
    {"name": "general-Q3", "domain": "general", "quant": "Q3",
     "path": "~/AdaptQuant/models/general-Q3.gguf"},
    {"name": "code-Q5",    "domain": "code",    "quant": "Q5",
     "path": "~/AdaptQuant/models/code-Q5.gguf"},
    {"name": "code-Q3",    "domain": "code",    "quant": "Q3",
     "path": "~/AdaptQuant/models/code-Q3.gguf"},
    {"name": "chat-Q5",    "domain": "chat",    "quant": "Q5",
     "path": "~/AdaptQuant/models/chat-Q5.gguf"},
    {"name": "chat-Q3",    "domain": "chat",    "quant": "Q3",
     "path": "~/AdaptQuant/models/chat-Q3.gguf"},
]

# ── Trimmed calibration files for perplexity ─────────────────────────────────
CALIB_FILES = {
    "general": "/tmp/wiki_ppl.txt",
    "code":    "/tmp/code_ppl.txt",
    "chat":    "/tmp/chat_ppl.txt",
}

# ── Test prompts per domain ───────────────────────────────────────────────────
PROMPTS = {
    "general": [
        "Explain the water cycle in simple terms.",
        "What causes inflation in an economy?",
        "Describe how vaccines work.",
        "What is the theory of relativity?",
        "Explain photosynthesis step by step.",
        "What is the difference between DNA and RNA?",
        "How does the human immune system work?",
        "Explain Newton's laws of motion.",
        "What is climate change and its causes?",
        "Describe the process of natural selection.",
    ],
    "code": [
        "Write a Python function to reverse a string.",
        "Explain what a binary search tree is.",
        "Write a function to check if a number is prime.",
        "What is the difference between a list and a tuple in Python?",
        "Explain recursion with a simple example.",
        "Write a Python function to find the factorial of a number.",
        "What is object oriented programming?",
        "Explain what an API is in simple terms.",
        "Write a function to sort a list using bubble sort.",
        "What is the difference between stack and queue?",
    ],
    "chat": [
        "Hi, how are you today?",
        "What do you enjoy doing in your free time?",
        "Can you recommend a good book to read?",
        "What is your favourite season and why?",
        "Tell me something interesting about space.",
        "What advice would you give to a new student?",
        "How do you stay motivated when things get hard?",
        "What is the most important skill to learn today?",
        "Can you tell me a short interesting story?",
        "What makes a good conversation?",
    ],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def prepare_calib_files():
    """Trim calibration files to 20 lines each for fast perplexity."""
    src = {
        "general": os.path.expanduser("~/AdaptQuant/data/calibration/general/wiki.txt"),
        "code":    os.path.expanduser("~/AdaptQuant/data/calibration/code/code.txt"),
        "chat":    os.path.expanduser("~/AdaptQuant/data/calibration/chat/chat.txt"),
    }
    for domain, src_path in src.items():
        dst = CALIB_FILES[domain]
        try:
            with open(src_path) as f:
                lines = []
                for _ in range(20):
                    try:
                        lines.append(next(f))
                    except StopIteration:
                        break
            with open(dst, "w") as f:
                f.writelines(lines)
            print(f"[bench] Prepared {domain} calib file ({len(lines)} lines).")
        except Exception as e:
            print(f"[bench] Error preparing {domain} calib file: {e}")

def start_server(model_path):
    cmd = [
        LLAMA_SERVER,
        "-m", os.path.expanduser(model_path),
        "--host", HOST,
        "--port", str(PORT),
        "--threads", str(THREADS),
        "--ctx-size", str(CTX_SIZE),
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            r = requests.get(f"http://{HOST}:{PORT}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except:
            pass
        time.sleep(1)
    proc.terminate()
    return None

def stop_server(proc):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()

def query(prompt):
    formatted = f"<|user|>\n{prompt}\n</s>\n<|assistant|>\n"
    try:
        r = requests.post(
            f"http://{HOST}:{PORT}/completion",
            json={
                "prompt":      formatted,
                "n_predict":   MAX_TOKENS,
                "temperature": 0.7,
                "stream":      False,
            },
            timeout=120,
        )
        data = r.json()
        timings = data.get("timings", {})
        return {
            "content":          data.get("content", ""),
            "tokens_predicted": data.get("tokens_predicted", 0),
            "prompt_tps":       round(timings.get("prompt_per_second", 0), 2),
            "gen_tps":          round(timings.get("predicted_per_second", 0), 2),
            "ttft_ms":          round(timings.get("prompt_ms", 0), 2),
        }
    except Exception as e:
        print(f"  [!] Query error: {e}")
        return None

def run_perplexity(model_path, calib_file):
    if not os.path.exists(calib_file):
        print(f"  [!] Calib file not found: {calib_file}")
        return None
    cmd = [
        LLAMA_PERP,
        "-m", os.path.expanduser(model_path),
        "-f", calib_file,
        "--threads", str(THREADS),
        "--ctx-size", str(CTX_SIZE),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "Final estimate: PPL" in line:
                ppl_str = line.split("=")[-1].strip().split()[0]
                return round(float(ppl_str), 4)
        print(f"  [!] PPL line not found in output. Last 5 lines:")
        for line in output.splitlines()[-5:]:
            print(f"      {line}")
    except Exception as e:
        print(f"  [!] Perplexity error: {e}")
    return None

def p90(values):
    if not values:
        return 0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * 0.90)
    return round(sorted_v[min(idx, len(sorted_v) - 1)], 2)

def append_csv(filepath, row):
    write_header = not os.path.exists(filepath)
    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ── Main benchmark loop ───────────────────────────────────────────────────────
def run_benchmarks():
    prepare_calib_files()
    summary = []
    total = len(MODELS)

    print("=" * 60)
    print("  AdaptQuant Benchmark Suite — Full Run")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"  Models : {total} | Prompts per model: {BENCH_PROMPTS}")
    print("=" * 60)

    for i, model in enumerate(MODELS):
        name   = model["name"]
        domain = model["domain"]
        quant  = model["quant"]
        path   = model["path"]

        print(f"\n[{i+1}/{total}] Benchmarking {name}...")

        if i > 0:
            print(f"  Cooling down {COOLDOWN_SEC}s...")
            for remaining in range(COOLDOWN_SEC, 0, -10):
                m = get_metrics()
                print(f"  {remaining}s remaining | Temp: {m['cpu_temp_c']}°C")
                time.sleep(10)

        proc = start_server(path)
        if not proc:
            print(f"  [!] Failed to start {name}. Skipping.")
            continue
        print(f"  Server ready.")

        # Warmup
        print(f"  Warming up ({WARMUP_PROMPTS} prompts)...")
        for w in range(WARMUP_PROMPTS):
            query(PROMPTS[domain][w])

        # Benchmark prompts
        gen_tps_list    = []
        ttft_ms_list    = []
        prompt_tps_list = []
        metrics_list    = []

        print(f"  Running {BENCH_PROMPTS} benchmark prompts...")
        for j in range(BENCH_PROMPTS):
            sys_metrics = get_metrics()
            result = query(PROMPTS[domain][j])
            if result:
                gen_tps_list.append(result["gen_tps"])
                ttft_ms_list.append(result["ttft_ms"])
                prompt_tps_list.append(result["prompt_tps"])
                metrics_list.append(sys_metrics)
                append_csv(INFERENCE_CSV, {
                    "timestamp":   datetime.now().isoformat(),
                    "model":       name,
                    "domain":      domain,
                    "quant":       quant,
                    "prompt_idx":  j,
                    "gen_tps":     result["gen_tps"],
                    "prompt_tps":  result["prompt_tps"],
                    "ttft_ms":     result["ttft_ms"],
                    "tokens":      result["tokens_predicted"],
                    "cpu_temp_c":  sys_metrics["cpu_temp_c"],
                    "ram_used_mb": sys_metrics["ram_used_mb"],
                    "ram_percent": sys_metrics["ram_percent"],
                })
                print(f"  [{j+1}/{BENCH_PROMPTS}] {result['gen_tps']} tok/s | "
                      f"{result['ttft_ms']}ms TTFT | {sys_metrics['cpu_temp_c']}°C")

        stop_server(proc)

        # Perplexity — all 3 domains
        print(f"  Running perplexity (3 domains)...")
        ppl_results = {}
        for test_domain, calib_file in CALIB_FILES.items():
            print(f"    PPL on {test_domain}...", end=" ", flush=True)
            ppl = run_perplexity(path, calib_file)
            ppl_results[test_domain] = ppl
            append_csv(PERPLEXITY_CSV, {
                "timestamp":   datetime.now().isoformat(),
                "model":       name,
                "domain":      domain,
                "quant":       quant,
                "test_domain": test_domain,
                "perplexity":  ppl,
            })
            print(f"{ppl}")

        # Model summary
        if gen_tps_list:
            model_summary = {
                "model":          name,
                "domain":         domain,
                "quant":          quant,
                "avg_gen_tps":    round(sum(gen_tps_list) / len(gen_tps_list), 2),
                "avg_ttft_ms":    round(sum(ttft_ms_list) / len(ttft_ms_list), 2),
                "p90_ttft_ms":    p90(ttft_ms_list),
                "avg_prompt_tps": round(sum(prompt_tps_list) / len(prompt_tps_list), 2),
                "avg_cpu_temp_c": round(sum(m["cpu_temp_c"] for m in metrics_list) / len(metrics_list), 2),
                "avg_ram_mb":     round(sum(m["ram_used_mb"] for m in metrics_list) / len(metrics_list), 2),
                "ppl_general":    ppl_results.get("general"),
                "ppl_code":       ppl_results.get("code"),
                "ppl_chat":       ppl_results.get("chat"),
            }
            summary.append(model_summary)
            print(f"\n  ── {name} summary ──")
            for k, v in model_summary.items():
                print(f"    {k}: {v}")

    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("  Benchmark complete!")
    print(f"  Finished: {datetime.now().isoformat()}")
    print(f"  Inference CSV : {INFERENCE_CSV}")
    print(f"  Perplexity CSV: {PERPLEXITY_CSV}")
    print(f"  Summary JSON  : {SUMMARY_JSON}")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmarks()