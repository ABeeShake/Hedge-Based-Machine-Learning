#!/usr/bin/env python3
"""record_runtime_env.py -- write overleaf/tables/runtime_env.csv (package versions and hardware).

Run once on the machine that executed the simulations (the GCP VM), inside its environment:
    python record_runtime_env.py
generate_results_macros.py turns the rows into \\Env...Version macros and \\EnvHardware.
"""
import csv
import importlib
import os
import platform
import subprocess

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "overleaf", "tables", "runtime_env.csv")


def _cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:
        return platform.processor() or platform.machine()


def _ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9)
    except (ValueError, OSError, AttributeError):
        return ""


def _gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"
    except Exception:
        pass
    return "none"


def main():
    rows = [("python", platform.python_version())]
    for pkg in ("numpy", "scipy", "pandas", "torch", "xgboost", "statsforecast", "neuralforecast", "torchdyn"):
        try:
            rows.append((pkg, importlib.import_module(pkg).__version__))
        except Exception:
            pass
    rows += [("os", f"{platform.system()} {platform.release()}"), ("cpu", _cpu_model()),
             ("cpu_cores", str(os.cpu_count())), ("ram_gb", str(_ram_gb())), ("gpu", _gpu()),
             ("engine_device", os.environ.get("FAST_ENGINE_DEVICE", "cpu"))]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["package", "version"])
        w.writerows(rows)
    print(f"Wrote {OUT}")
    for k, v in rows:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
