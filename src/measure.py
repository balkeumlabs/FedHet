"""On-device cost measurement: wall-clock, CPU time, and real package energy.

Energy is read from the Linux powercap RAPL interface (same method as Balkeum's
CEM26 study on this hardware). On this AMD Ryzen the package domain is exposed at
/sys/class/powercap/intel-rapl:*/energy_uj. If RAPL is unreadable we fall back to
a labeled TDP estimate so the harness still runs everywhere.
"""
from __future__ import annotations
import glob
import os
import time
from dataclasses import dataclass

TDP_WATTS = 65.0  # AMD Ryzen 5 5500GT, used only if RAPL is unavailable.


def _rapl_domains() -> list[str]:
    out = []
    for d in glob.glob("/sys/class/powercap/intel-rapl:*"):
        # top-level package domains only (skip subdomains intel-rapl:0:0)
        if d.count(":") == 1 and os.path.exists(os.path.join(d, "energy_uj")):
            out.append(d)
    return sorted(out)


def _read_energy_uj(domains) -> float | None:
    total = 0.0
    ok = False
    for d in domains:
        try:
            with open(os.path.join(d, "energy_uj")) as f:
                total += float(f.read())
            ok = True
        except OSError:
            return None
    return total if ok else None


@dataclass
class CostReport:
    wall_s: float
    cpu_s: float
    energy_j: float
    energy_source: str  # "rapl" | "tdp_estimate"


def measure(fn, *args, **kwargs):
    """Run fn(*args, **kwargs); return (result, CostReport)."""
    domains = _rapl_domains()
    e0 = _read_energy_uj(domains) if domains else None
    t0, c0 = time.perf_counter(), time.process_time()
    result = fn(*args, **kwargs)
    t1, c1 = time.perf_counter(), time.process_time()
    e1 = _read_energy_uj(domains) if domains else None

    wall = t1 - t0
    cpu = c1 - c0
    if e0 is not None and e1 is not None and e1 >= e0:
        energy_j = (e1 - e0) / 1e6
        src = "rapl"
    else:
        energy_j = TDP_WATTS * wall
        src = "tdp_estimate"
    return result, CostReport(wall, cpu, energy_j, src)


def model_bytes(dim: int, dtype_bytes: int = 4) -> int:
    """Uplink payload per home per round: weight vector + bias (float32)."""
    return (dim + 1) * dtype_bytes
