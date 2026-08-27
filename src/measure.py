"""On-device cost measurement: wall-clock, CPU time, and real package energy.

Energy is read from the Linux powercap RAPL interface (the same method as Balkeum's
CEM26 study on this hardware). The package domain is exposed at
``/sys/class/powercap/intel-rapl:*/energy_uj`` (the ``intel-rapl`` driver also backs
AMD Zen ``amd_energy``-style counters on recent kernels).

**Reproducibility note.** On most current distributions ``energy_uj`` is mode 0400
root-only, a mitigation for the PLATYPUS side-channel (CVE-2020-8694). If the file
cannot be read, this module falls back to a *labeled TDP estimate* and prints a
prominent warning; every result carries an ``energy_source`` field
(``"rapl"`` | ``"tdp_estimate"``) so an estimated number can never be mistaken for a
measured one. To reproduce the measured energy figures, grant read access first:

    sudo chmod o+r /sys/class/powercap/intel-rapl:*/energy_uj

See ``README.md`` ("Reproducing the energy measurement") for the full procedure.
"""
from __future__ import annotations
import glob
import os
import sys
import time
from dataclasses import dataclass

TDP_WATTS = 65.0  # AMD Ryzen 5 5500GT rated TDP; used only if RAPL is unavailable.

RAPL_GLOB = "/sys/class/powercap/intel-rapl:*"
_WARNED = False


def _rapl_domains() -> list[str]:
    """Top-level RAPL package domains (skip subdomains such as intel-rapl:0:0)."""
    out = []
    for d in glob.glob(RAPL_GLOB):
        if d.count(":") == 1 and os.path.exists(os.path.join(d, "energy_uj")):
            out.append(d)
    return sorted(out)


def rapl_status() -> tuple[bool, str]:
    """Return (readable, human-readable reason). Used for reporting, not control."""
    domains = _rapl_domains()
    if not domains:
        return False, (f"no RAPL package domain found under {RAPL_GLOB} "
                       "(not a Linux powercap-capable host, or driver not loaded)")
    for d in domains:
        path = os.path.join(d, "energy_uj")
        try:
            with open(path) as f:
                f.read()
        except PermissionError:
            return False, (f"{path} is not readable by this user (root-only 0400 is "
                           "the default mitigation for CVE-2020-8694). Run: "
                           f"sudo chmod o+r {RAPL_GLOB}/energy_uj")
        except OSError as e:
            return False, f"{path} unreadable: {e}"
    return True, f"RAPL package domains readable: {', '.join(domains)}"


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


def _warn_once(reason: str) -> None:
    global _WARNED
    if _WARNED:
        return
    _WARNED = True
    print(
        "\n" + "!" * 78 +
        "\n!! ENERGY IS ESTIMATED, NOT MEASURED."
        f"\n!! Reason: {reason}"
        f"\n!! Falling back to a {TDP_WATTS:.0f} W TDP x wall-clock estimate; every"
        "\n!! reported energy value is tagged energy_source=\"tdp_estimate\" and is"
        "\n!! NOT comparable to the RAPL-measured figures in results/results.json."
        "\n" + "!" * 78 + "\n",
        file=sys.stderr,
    )


@dataclass
class CostReport:
    wall_s: float
    cpu_s: float
    energy_j: float
    energy_source: str  # "rapl" | "tdp_estimate"


def measure(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)``; return ``(result, CostReport)``."""
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
        _warn_once(rapl_status()[1])
        energy_j = TDP_WATTS * wall
        src = "tdp_estimate"
    return result, CostReport(wall, cpu, energy_j, src)


def model_bytes(dim: int, dtype_bytes: int = 4) -> int:
    """Uplink payload per home per round: weight vector + bias, as float32."""
    return (dim + 1) * dtype_bytes
