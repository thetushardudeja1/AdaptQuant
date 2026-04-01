# daemon.py — main orchestrator that ties everything together

import time
import os
import sys

sys.path.insert(0, os.path.expanduser("~/AdaptQuant"))

import orchestrator.hot_swap as hs
from orchestrator.policy import decide, get_stress
from scripts.monitor import get_metrics
from scripts.logger import log_inference, log_switch

# Model paths
MODELS = {
    ("general", "Q5"): "~/AdaptQuant/models/general-Q5.gguf",
    ("general", "Q3"): "~/AdaptQuant/models/general-Q3.gguf",
    ("code",    "Q5"): "~/AdaptQuant/models/code-Q5.gguf",
    ("code",    "Q3"): "~/AdaptQuant/models/code-Q3.gguf",
    ("chat",    "Q5"): "~/AdaptQuant/models/chat-Q5.gguf",
    ("chat",    "Q3"): "~/AdaptQuant/models/chat-Q3.gguf",
}

# Starting state
current_domain = "general"
current_quant  = "Q5"

def get_current_model_path():
    return MODELS[(current_domain, current_quant)]

def run(start_domain="general", start_quant="Q5"):
    global current_domain, current_quant
    current_domain = start_domain
    current_quant  = start_quant

    print(f"[daemon] Starting with {current_domain}-{current_quant}")
    hs.initialize(get_current_model_path())

    print("[daemon] Running. Type your prompt and press Enter. Ctrl+C to quit.\n")

    try:
        while True:
            # Get user prompt
            try:
                prompt = input("You: ").strip()
            except EOFError:
                break
            if not prompt:
                continue

            # Read system metrics before inference
            metrics = get_metrics()

            # Compute stress score
            stress = get_stress(metrics['cpu_temp_c'])

            # Run inference
            result = hs.query(prompt, max_tokens=150)
            if not result:
                print("[daemon] No response from model.")
                continue

            print(f"\nAssistant ({current_domain}-{current_quant}): {result['content']}")
            print(f"[{result['gen_tps']} tok/s | "
                  f"{metrics['cpu_temp_c']}°C | "
                  f"RAM: {metrics['ram_percent']}% | "
                  f"stress: {stress}]\n")

            # Log inference
            log_inference(
                model_name  = f"{current_domain}-{current_quant}",
                domain      = current_domain,
                quant_level = current_quant,
                prompt_tps  = result["prompt_tps"],
                gen_tps     = result["gen_tps"],
                ttft_ms     = result["ttft_ms"],
                ram_mb      = metrics["ram_used_mb"],
                cpu_temp    = metrics["cpu_temp_c"],
            )

            # Check switching policy
            new_quant, reason = decide(
                cpu_temp      = metrics["cpu_temp_c"],
                ram_percent   = metrics["ram_percent"],
                gen_tps       = result["gen_tps"],
                current_quant = current_quant,
            )

            if new_quant:
                print(f"[daemon] Switching {current_quant} → {new_quant} | "
                      f"reason: {reason}")
                new_path   = MODELS[(current_domain, new_quant)]
                latency_ms = hs.swap(new_path)

                log_switch(
                    from_model        = f"{current_domain}-{current_quant}",
                    to_model          = f"{current_domain}-{new_quant}",
                    reason            = reason,
                    switch_latency_ms = latency_ms,
                    cpu_temp          = metrics["cpu_temp_c"],
                    ram_mb            = metrics["ram_used_mb"],
                )

                current_quant = new_quant
                print(f"[daemon] Now running {current_domain}-{current_quant}\n")

    except KeyboardInterrupt:
        print("\n[daemon] Shutting down...")

    finally:
        hs.shutdown()
        print("[daemon] Done.")

if __name__ == "__main__":
    run(start_domain="general", start_quant="Q5")