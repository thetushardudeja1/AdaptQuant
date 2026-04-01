# policy.py — PID-inspired thermal controller for quantization switching
# Treats thermal stability as the controlled variable
# Quantization level is the actuator

import time
from collections import deque

# ── PID targets and limits ────────────────────────────────────────────────────
TARGET_TEMP     = 60.0   # °C — ideal operating temperature
TEMP_MAX        = 72.0   # °C — hard ceiling, always downgrade
TEMP_SAFE       = 55.0   # °C — safe to upgrade back

RAM_MAX         = 85.0   # % — RAM ceiling
TPS_MIN         = 8.0    # tok/s — minimum acceptable speed

# ── PID gains (tuned from benchmark data) ────────────────────────────────────
Kp = 0.6    # proportional — reacts to current deviation
Ki = 0.15   # integral — reacts to sustained heat buildup
Kd = 0.8    # derivative — reacts to rate of temperature rise

# ── Stress thresholds ─────────────────────────────────────────────────────────
STRESS_DOWNGRADE = 60.0  # stress score above this → switch to Q3
STRESS_UPGRADE   = 30.0  # stress score below this → switch back to Q5

# ── Internal PID state ────────────────────────────────────────────────────────
_error_history  = deque(maxlen=30)   # last 30 seconds of errors
_last_error     = 0.0
_last_time      = None
_integral       = 0.0

def reset():
    """Reset PID state — call when switching models."""
    global _error_history, _last_error, _last_time, _integral
    _error_history.clear()
    _last_error  = 0.0
    _last_time   = None
    _integral    = 0.0

def _compute_pid(current_temp):
    """
    Compute PID stress score from current temperature.
    Returns stress score 0–100.
    """
    global _last_error, _last_time, _integral

    now   = time.time()
    error = current_temp - TARGET_TEMP  # positive = too hot

    # Time delta
    if _last_time is None:
        dt = 1.0
    else:
        dt = max(now - _last_time, 0.001)
    _last_time = now

    # Proportional
    P = Kp * error

    # Integral — accumulates sustained heat, clamp to prevent windup
    _integral += error * dt
    _integral  = max(-50.0, min(50.0, _integral))
    I = Ki * _integral

    # Derivative — rate of temperature change (rising fast = act early)
    D = Kd * (error - _last_error) / dt
    _last_error = error

    # Raw PID output
    raw = P + I + D

    # Normalise to 0–100 stress score
    stress = max(0.0, min(100.0, 50.0 + raw))
    return round(stress, 2), round(P, 3), round(I, 3), round(D, 3)

def decide(cpu_temp, ram_percent, gen_tps, current_quant):
    """
    Main decision function.
    Returns: (new_quant, reason) or (None, None) if no switch needed.
    """
    # ── Hard overrides (bypass PID) ──
    if cpu_temp >= TEMP_MAX:
        if current_quant == "Q5":
            reset()
            return "Q3", f"hard_temp_ceiling ({cpu_temp}°C >= {TEMP_MAX})"
        return None, None

    if ram_percent >= RAM_MAX:
        if current_quant == "Q5":
            reset()
            return "Q3", f"hard_ram_ceiling ({ram_percent}% >= {RAM_MAX}%)"
        return None, None

    if gen_tps < TPS_MIN:
        if current_quant == "Q5":
            reset()
            return "Q3", f"hard_tps_floor ({gen_tps} tok/s < {TPS_MIN})"
        return None, None

    # ── PID stress score ──
    stress, P, I, D = _compute_pid(cpu_temp)

    # Downgrade if stress too high
    if stress >= STRESS_DOWNGRADE:
        if current_quant == "Q5":
            reset()
            return "Q3", (f"pid_stress_high (score={stress} | "
                         f"P={P} I={I} D={D} | temp={cpu_temp}°C)")

    # Upgrade if stress low enough and system recovered
    if stress <= STRESS_UPGRADE and cpu_temp < TEMP_SAFE:
        if current_quant == "Q3":
            reset()
            return "Q5", (f"pid_stress_low (score={stress} | "
                         f"P={P} I={I} D={D} | temp={cpu_temp}°C)")

    return None, None

def get_stress(cpu_temp):
    """Public helper — returns current stress score without making a decision."""
    stress, P, I, D = _compute_pid(cpu_temp)
    return stress

if __name__ == "__main__":
    # Simulate a temperature sequence and show PID decisions
    import math

    print(f"{'Time':>5} {'Temp':>6} {'Stress':>8} {'P':>7} {'I':>7} {'D':>7} {'Decision'}")
    print("-" * 75)

    scenarios = (
        # Gradual heat buildup
        [(55 + i * 0.5, "Q5") for i in range(20)] +
        # Spike
        [(65 + i * 1.5, "Q5") for i in range(8)] +
        # Cooling down on Q3
        [(75 - i * 1.2, "Q3") for i in range(20)]
    )

    for t, (temp, quant) in enumerate(scenarios):
        stress, P, I, D = _compute_pid(temp)
        new_quant, reason = decide(temp, 50.0, 15.0, quant)
        decision = f"→ {new_quant} ({reason})" if new_quant else "hold"
        print(f"{t:>5} {temp:>6.1f} {stress:>8.2f} {P:>7.3f} {I:>7.3f} "
              f"{D:>7.3f} {decision}")
        time.sleep(0.05)