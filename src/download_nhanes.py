"""Download the raw NHANES files used in this study from the CDC public server.

NHANES data are released by the U.S. Centers for Disease Control and Prevention
(CDC) National Center for Health Statistics and are fully public (no credentialing
required). We use real survey data only -- no synthetic data is generated and no
record is perturbed.

Files (consistent variables across the 2015-2016 "_I" and 2017-2018 "_J" cycles):
    DEMO    demographics      (age, sex)
    BMX     body measures     (BMI, weight, waist)
    BPX     blood pressure    (systolic, diastolic, pulse)
    GHB     glycohemoglobin   (HbA1c -- CGM/Tier-3 glycemic signal)
    SMQ     smoking           (current-smoker derivation)
    DIQ     diabetes          (physician-diagnosed-diabetes label)

URL pattern (verified June 2026):
    https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{startyear}/DataFiles/{NAME}{SUF}.xpt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import requests

BASE = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}{suf}.xpt"

# cycle label -> (path year, file suffix)
CYCLES = {
    "2015-2016": ("2015", "_I"),
    "2017-2018": ("2017", "_J"),
}

# component base names (suffix appended per cycle)
COMPONENTS = ["DEMO", "BMX", "BPX", "GHB", "SMQ", "DIQ"]

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def download(url: str, dest: Path, timeout: int = 120) -> bool:
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  cached  {dest.name} ({dest.stat().st_size/1e6:.1f} MB)")
        return True
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            ctype = r.headers.get("content-type", "")
            tmp = dest.with_suffix(".part")
            n = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    n += len(chunk)
            # XPT files begin with "HEADER RECORD"; reject HTML error pages.
            head = open(tmp, "rb").read(32)
            if not head.startswith(b"HEADER RECORD") or n < 1024:
                tmp.unlink(missing_ok=True)
                print(f"  FAIL    {dest.name}: not an XPT (ctype={ctype}, {n} B)")
                return False
            tmp.rename(dest)
            print(f"  ok      {dest.name} ({n/1e6:.1f} MB)")
            return True
    except requests.RequestException as e:
        print(f"  FAIL    {dest.name}: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Download raw NHANES XPT files.")
    ap.add_argument("--cycles", nargs="+", default=list(CYCLES),
                    choices=list(CYCLES), help="NHANES cycles to fetch.")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ok = True
    for cyc in args.cycles:
        year, suf = CYCLES[cyc]
        print(f"[{cyc}]")
        for comp in COMPONENTS:
            url = BASE.format(year=year, name=comp, suf=suf)
            dest = RAW_DIR / f"{comp}{suf}.xpt"
            ok &= download(url, dest)
    if not ok:
        print("\nOne or more files failed to download.", file=sys.stderr)
        return 1
    print(f"\nAll files in {RAW_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
