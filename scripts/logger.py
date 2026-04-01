import csv
import json
import os
from datetime import datetime

INFERENCE_LOG = os.path.expanduser("~/AdaptQuant/results/inference_log.csv")
SWITCH_LOG = os.path.expanduser("~/AdaptQuant/results/switch_events.jsonl")

os.makedirs(os.path.expanduser("~/AdaptQuant/results"), exist_ok=True)

def log_inference(model_name, domain, quant_level, prompt_tps, gen_tps, ttft_ms, ram_mb, cpu_temp):
    row = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "domain": domain,
        "quant_level": quant_level,
        "prompt_tps": prompt_tps,
        "gen_tps": gen_tps,
        "ttft_ms": ttft_ms,
        "ram_mb": ram_mb,
        "cpu_temp_c": cpu_temp,
    }
    write_header = not os.path.exists(INFERENCE_LOG)
    with open(INFERENCE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"[logger] Inference logged: {model_name} | {gen_tps} tok/s | {cpu_temp}°C")

def log_switch(from_model, to_model, reason, switch_latency_ms, cpu_temp, ram_mb):
    event = {
        "timestamp": datetime.now().isoformat(),
        "from_model": from_model,
        "to_model": to_model,
        "reason": reason,
        "switch_latency_ms": switch_latency_ms,
        "cpu_temp_c": cpu_temp,
        "ram_mb": ram_mb,
    }
    with open(SWITCH_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")
    print(f"[logger] Switch logged: {from_model} → {to_model} | reason: {reason} | latency: {switch_latency_ms}ms")