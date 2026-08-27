#!/usr/bin/env python3
"""Verify a local reproduction against the published results committed in this repo.

Usage (from the repository root, after `python src/run_experiment.py`):

    python scripts/verify_results.py                     # compare results/results.json
    python scripts/verify_results.py --candidate PATH    # compare some other run
    python scripts/verify_results.py --reference PATH    # against another reference
    python scripts/verify_results.py --tol 1e-9          # tighten/loosen tolerance

The reference is `results/results.json` as committed, which is the run reported in
the paper. This script checks the numbers a reader would actually cite:

  * headline test AUROC / AUPRC / Brier / accuracy for all four methods,
  * the derived gaps and the recovery of tier-aware FL over intersection FL,
  * rounds-to-converge for each federated method,
  * per-tier marginal contribution and the learned reward ladder,
  * communication accounting (exact integers),
  * the cohort and configuration the run used.

Energy is deliberately NOT compared. RAPL package energy depends on the host CPU
and on whether /sys/class/powercap is readable at all; a reproduction on different
hardware is expected to differ. The script reports the energy source of both runs
so a `tdp_estimate` reproduction is never silently compared to a `rapl` reference.

Exit status is 0 if every checked quantity matches within tolerance, 1 otherwise.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT = ROOT / "results" / "results.json"

METHODS = ["centralized", "intersection_fl", "naive_union_fl", "tieraware_fl"]
FL_METHODS = ["intersection_fl", "naive_union_fl", "tieraware_fl"]
METRIC_KEYS = ["auroc", "auprc", "brier", "acc"]

# A round "converges" once its test AUROC is within this many AUROC points of the
# value the method reaches at the final round. 0.005 = 0.5 percentage points.
CONVERGENCE_TOL = 0.005


def rounds_to_converge(history: list[float], tol: float = CONVERGENCE_TOL) -> int:
    """First 1-indexed round within `tol` AUROC of the method's final-round value.

    This is the definition behind the "rounds to converge" column in the README
    and the paper. It is computed from the per-round test AUROC curve stored in
    results.json under "histories", so it is fully derivable from committed data.
    """
    if not history:
        return 0
    final = history[-1]
    for i, auc in enumerate(history):
        if auc >= final - tol:
            return i + 1
    return len(history)


class Checker:
    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.failures: list[str] = []
        self.checks = 0

    def close(self, label: str, got: float, want: float) -> None:
        self.checks += 1
        if abs(got - want) > self.tol:
            self.failures.append(
                f"{label}: got {got!r}, expected {want!r} "
                f"(|diff|={abs(got - want):.3g} > tol={self.tol:g})")
            status = "FAIL"
        else:
            status = "ok"
        print(f"  [{status:4s}] {label:52s} {got:.6f}  (ref {want:.6f})")

    def equal(self, label: str, got, want) -> None:
        self.checks += 1
        if got != want:
            self.failures.append(f"{label}: got {got!r}, expected {want!r}")
            status = "FAIL"
        else:
            status = "ok"
        print(f"  [{status:4s}] {label:52s} {got!r}  (ref {want!r})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", type=Path, default=DEFAULT,
                    help="results.json produced by your run (default: %(default)s)")
    ap.add_argument("--reference", type=Path, default=None,
                    help="published results.json to compare against "
                         "(default: the git-committed results/results.json)")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="absolute tolerance for float comparisons "
                         "(default: %(default)s)")
    args = ap.parse_args()

    ref_path = args.reference or DEFAULT
    if not args.candidate.exists():
        print(f"candidate not found: {args.candidate}\n"
              "Run `python src/run_experiment.py` first.", file=sys.stderr)
        return 1
    if not ref_path.exists():
        print(f"reference not found: {ref_path}", file=sys.stderr)
        return 1

    cand = json.loads(args.candidate.read_text())
    ref = json.loads(ref_path.read_text())
    if args.candidate.resolve() == ref_path.resolve():
        print(f"NOTE: candidate and reference are the same file ({ref_path}).\n"
              "      This run is a self-consistency check of the published results.\n")

    ck = Checker(args.tol)

    print("Cohort and configuration")
    for key in ("n_train", "n_test", "label_prevalence", "features",
                "seed", "homes", "alpha", "rounds", "local_epochs", "lr", "l2",
                "cen_epochs", "tier_counts", "tier_mix"):
        if key in ref["config"]:
            ck.equal(f"config.{key}", cand["config"].get(key), ref["config"][key])

    print("\nHeadline metrics")
    for m in METHODS:
        for k in METRIC_KEYS:
            ck.close(f"metrics.{m}.{k}", cand["metrics"][m][k], ref["metrics"][m][k])

    print("\nDerived claims (percentage points of AUROC)")

    def gaps(d):
        auc = {m: d["metrics"][m]["auroc"] for m in METHODS}
        return {
            "intersection gap vs centralized":
                (auc["centralized"] - auc["intersection_fl"]) * 100,
            "tier-aware gap vs centralized":
                (auc["centralized"] - auc["tieraware_fl"]) * 100,
            "tier-aware recovery over intersection":
                (auc["tieraware_fl"] - auc["intersection_fl"]) * 100,
        }

    gc, gr = gaps(cand), gaps(ref)
    for label in gc:
        ck.close(label, gc[label], gr[label])

    print(f"\nRounds to converge (within {CONVERGENCE_TOL * 100:.1f} pp of the "
          "final-round AUROC)")
    for m in FL_METHODS:
        ck.equal(f"rounds_to_converge.{m}",
                 rounds_to_converge(cand["histories"][m]),
                 rounds_to_converge(ref["histories"][m]))

    print("\nPer-tier contribution and reward ladder")
    ck.close("contribution.profile_auc",
             cand["contribution"]["profile_auc"], ref["contribution"]["profile_auc"])
    for tc, tr in zip(cand["contribution"]["tiers"],
                      ref["contribution"]["tiers"], strict=True):
        ck.equal(f"tier {tr['tier']} name", tc["name"], tr["name"])
        for k in ("cum_auc", "marginal_auc_gain", "learned_reward_share"):
            ck.close(f"tier {tr['tier']}.{k}", tc[k], tr[k])

    print("\nCommunication accounting (exact)")
    for k, v in ref["comms"].items():
        ck.equal(f"comms.{k}", cand["comms"].get(k), v)

    print("\nScaling sweep")
    ck.equal("sweep.n_homes", cand["sweep"]["n_homes"], ref["sweep"]["n_homes"])
    for m in FL_METHODS:
        pairs = zip(cand["sweep"][m], ref["sweep"][m], strict=True)
        for n, (a, b) in zip(ref["sweep"]["n_homes"], pairs, strict=True):
            ck.close(f"sweep.{m}.n_homes={n}", a, b)

    print("\nEnergy (reported, NOT compared -- hardware dependent)")
    for m in METHODS:
        c, r = cand["costs"][m], ref["costs"][m]
        print(f"  [info] {m:20s} "
              f"candidate {c['energy_j']:8.3f} J ({c['energy_source']})"
              f"   reference {r['energy_j']:8.3f} J ({r['energy_source']})")
    if cand["platform"]["energy_source"] != "rapl":
        print("\n  NOTE: this reproduction did not read RAPL, so its energy figures are\n"
              "        TDP estimates and are not comparable to the published\n"
              "        measurement.\n"
              "        See README, 'Reproducing the energy measurement'.")

    print("\n" + "=" * 72)
    if ck.failures:
        print(f"FAILED: {len(ck.failures)} of {ck.checks} checks did not match.\n")
        for f in ck.failures:
            print(f"  - {f}")
        return 1
    print(f"PASSED: all {ck.checks} checks match the published results "
          f"(tolerance {args.tol:g}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
