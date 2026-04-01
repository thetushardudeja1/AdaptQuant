# hot_swap.py — silent model switching via llama.cpp server mode

import subprocess
import time
import os
import requests

LLAMA_SERVER = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
HOST = "127.0.0.1"
PORT_ACTIVE  = 8080
PORT_STANDBY = 8081

active_process  = None
standby_process = None
active_port     = PORT_ACTIVE
standby_port    = PORT_STANDBY

def start_server(model_path, port, threads=4):
    """Start llama-server on given port, return process handle."""
    cmd = [
        LLAMA_SERVER,
        "-m", os.path.expanduser(model_path),
        "--host", HOST,
        "--port", str(port),
        "--threads", str(threads),
        "--ctx-size", "2048",
        "--log-disable",
    ]
    print(f"[hot_swap] Starting server on port {port}: {os.path.basename(model_path)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return process

def wait_until_ready(port, timeout=60):
    """Poll /health until server responds or timeout."""
    url = f"http://{HOST}:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                print(f"[hot_swap] Server on port {port} is ready.")
                return True
        except:
            pass
        time.sleep(1)
    print(f"[hot_swap] ERROR: Server on port {port} did not start in {timeout}s.")
    return False

def kill_server(process, port):
    """Gracefully kill a server process."""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except:
            process.kill()
        print(f"[hot_swap] Server on port {port} killed.")

def get_active_port():
    return active_port

def initialize(model_path):
    """Start the first model on active port."""
    global active_process, active_port
    active_process = start_server(model_path, PORT_ACTIVE)
    if wait_until_ready(PORT_ACTIVE):
        active_port = PORT_ACTIVE
        print(f"[hot_swap] Initialized on port {PORT_ACTIVE}")
        return True
    return False

def swap(new_model_path):
    """
    Hot-swap to new model silently:
    1. Start new model on standby port
    2. Wait until ready
    3. Flip active to standby
    4. Kill old model
    Returns switch latency in ms.
    """
    global active_process, standby_process
    global active_port, standby_port

    t_start = time.time()

    # Step 1 — start new model on standby port
    standby_process = start_server(new_model_path, standby_port)

    # Step 2 — wait until ready
    ready = wait_until_ready(standby_port)
    if not ready:
        standby_process.terminate()
        return None

    # Step 3 — flip ports
    old_process = active_process
    old_port = active_port

    active_process = standby_process
    active_port = standby_port
    standby_port = old_port

    # Step 4 — kill old model
    kill_server(old_process, old_port)

    latency_ms = round((time.time() - t_start) * 1000)
    print(f"[hot_swap] Swap complete in {latency_ms}ms. Now serving on port {active_port}")
    return latency_ms

def query(prompt, max_tokens=200):
    """Send a prompt to the active server and return response."""
    url = f"http://{HOST}:{active_port}/completion"
    formatted = f"<|user|>\n{prompt}\n</s>\n<|assistant|>\n"
    payload = {
        "prompt": formatted,
        "n_predict": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        data = r.json()
        return {
            "content": data.get("content", ""),
            "tokens_predicted": data.get("tokens_predicted", 0),
            "prompt_tps": round(data["timings"]["prompt_per_second"], 2),
            "gen_tps": round(data["timings"]["predicted_per_second"], 2),
            "ttft_ms": round(data["timings"]["prompt_ms"], 2),
        }
    except Exception as e:
        print(f"[hot_swap] Query error: {e}")
        return None

def shutdown():
    """Kill all servers cleanly."""
    kill_server(active_process, active_port)
    kill_server(standby_process, standby_port)