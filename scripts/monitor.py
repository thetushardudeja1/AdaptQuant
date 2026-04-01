import psutil
import time
import csv
import os
from datetime import datetime

LOG_PATH = os.path.expanduser("~/AdaptQuant/results/system_metrics.csv")

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return round(int(f.read()) / 1000, 2)
    except:
        return 0.0

def get_metrics():
    return {
        "timestamp": datetime.now().isoformat(),
        "cpu_temp_c": get_cpu_temp(),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": round(psutil.virtual_memory().used / 1024 / 1024, 2),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_mb": round(psutil.virtual_memory().available / 1024 / 1024, 2),
    }

def start_logging(interval_sec=1):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(LOG_PATH)

    print(f"[monitor] Logging to {LOG_PATH} every {interval_sec}s. Ctrl+C to stop.")

    with open(LOG_PATH, "a", newline="") as f:
        writer = None
        try:
            while True:
                metrics = get_metrics()
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=metrics.keys())
                    if write_header:
                        writer.writeheader()
                writer.writerow(metrics)
                f.flush()
                print(f"[{metrics['timestamp']}] "
                      f"Temp: {metrics['cpu_temp_c']}°C | "
                      f"CPU: {metrics['cpu_percent']}% | "
                      f"RAM: {metrics['ram_used_mb']}MB ({metrics['ram_percent']}%)")
                time.sleep(interval_sec)
        except KeyboardInterrupt:
            print("\n[monitor] Stopped.")

if __name__ == "__main__":
    start_logging()