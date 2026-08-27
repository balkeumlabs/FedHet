"""End-to-end experiment: federated diabetes-risk scoring under device-tier heterogeneity.

Produces results/*.json, results/*.csv and figures/*.png. CPU-only; runs on a
consumer edge node (developed on an AMD Ryzen 5 5500GT, 62 GB RAM, no GPU used).
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import train_test_split

import config as C
from partition import make_homes
from federated import (federated_scaler, apply_scaler, train_federated,
                       train_centralized)
from measure import measure, model_bytes, CostReport
from contribution import tier_contributions

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "nhanes_cohort.csv"
RESULTS = ROOT / "results"
FIGS = ROOT / "figures"

FL_METHODS = ["intersection_fl", "naive_union_fl", "tieraware_fl"]
PRETTY = {"centralized": "Centralized\n(upper bound)",
          "intersection_fl": "Intersection FL\n(Tier-1 only)",
          "naive_union_fl": "Naive union FL",
          "tieraware_fl": "Tier-aware FL\n(ours)"}


def metrics(y, p) -> dict:
    return {"auroc": float(roc_auc_score(y, p)),
            "auprc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "acc": float(((p >= 0.5).astype(int) == y).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--homes", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.5, help="Dirichlet non-IID")
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--local-epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--cen-epochs", type=int, default=400)
    ap.add_argument("--out-dir", type=Path, default=RESULTS,
                    help="where to write results.json / comparison.csv "
                         "(default: %(default)s, which overwrites the published "
                         "artifacts; point elsewhere to keep them)")
    ap.add_argument("--fig-dir", type=Path, default=FIGS,
                    help="where to write the figures (default: %(default)s)")
    args = ap.parse_args()
    out_dir, fig_dir = args.out_dir, args.fig_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Tier adoption mix (mirrors product: most homes Core, fewer Total Vital).
    tier_mix = {1: 0.5, 2: 0.3, 3: 0.2}

    df = pd.read_csv(DATA)
    feat_order = C.all_features()
    X = df[feat_order].to_numpy(float)
    y = df[C.LABEL].to_numpy(float)

    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # --- homes + federated standardization (stats only leave the home) ---
    homes = make_homes(Xtr_raw, ytr, feat_order, args.homes, args.alpha,
                       tier_mix, args.seed)
    mean, std = federated_scaler(homes, feat_order)
    for h in homes:
        h.X = apply_scaler(h.X, h.mask, mean, std)
    Xte = (Xte_raw - mean) / std
    Xtr = (Xtr_raw - mean) / std

    def eval_fn(model, restrict):
        """Per-round test AUROC, for the convergence curve (Fig. 2)."""
        return roc_auc_score(yte, model.proba(Xte))

    print(f"n_train={len(ytr)} n_test={len(yte)} homes={len(homes)} "
          f"features={len(feat_order)}")
    tier_counts = {t: sum(h.tier == t for h in homes) for t in (1, 2, 3)}
    shard_sizes = [len(h.y) for h in homes]
    print(f"tier counts={tier_counts} shard size "
          f"min/med/max={min(shard_sizes)}/{int(np.median(shard_sizes))}/{max(shard_sizes)}")

    # --- main comparison ---
    results = {}
    costs: dict[str, CostReport] = {}

    cen_model, costs["centralized"] = measure(
        train_centralized, Xtr, ytr,
        epochs=args.cen_epochs, lr=args.lr, l2=args.l2)
    results["centralized"] = metrics(yte, cen_model.proba(Xte))
    results["centralized"]["history"] = []

    histories = {}
    for method in FL_METHODS:
        (model, restrict, hist), costs[method] = measure(
            train_federated, homes, feat_order, method=method,
            rounds=args.rounds, local_epochs=args.local_epochs, lr=args.lr,
            l2=args.l2, clients_per_round=None, seed=args.seed, eval_fn=eval_fn)
        results[method] = metrics(yte, model.proba(Xte))
        results[method]["history"] = hist
        histories[method] = hist

    # --- communication accounting ---
    D = len(feat_order)
    per_round_bytes = model_bytes(D) * len(homes)  # uplink, all homes
    comms = {"model_params": D + 1,
             "bytes_per_home_per_round": model_bytes(D),
             "bytes_per_round_all_homes": per_round_bytes,
             "total_bytes_tieraware": per_round_bytes * args.rounds,
             "raw_data_egress_bytes": 0}

    # --- contribution / reward ladder (FLAI tie-in) ---
    contrib = tier_contributions(Xtr, ytr, Xte, yte, feat_order,
                                 epochs=args.cen_epochs, lr=args.lr, l2=args.l2)

    # --- scaling sweep: AUROC vs number of homes ---
    sweep = {m: [] for m in FL_METHODS}
    sweep_homes = [20, 40, 60, 100, 150]
    for nh in sweep_homes:
        hs = make_homes(Xtr_raw, ytr, feat_order, nh, args.alpha, tier_mix,
                        args.seed)
        mn, sd = federated_scaler(hs, feat_order)
        for h in hs:
            h.X = apply_scaler(h.X, h.mask, mn, sd)
        Xte_s = (Xte_raw - mn) / sd
        for m in FL_METHODS:
            mod, _, _ = train_federated(
                hs, feat_order, method=m, rounds=args.rounds,
                local_epochs=args.local_epochs, lr=args.lr, l2=args.l2,
                clients_per_round=None, seed=args.seed)
            sweep[m].append(float(roc_auc_score(yte, mod.proba(Xte_s))))

    # --- assemble + save ---
    # Output paths are a runtime choice, not part of the experiment's definition,
    # so they are excluded from the recorded config.
    cfg = {k: v for k, v in vars(args).items() if k not in ("out_dir", "fig_dir")}
    out = {
        "config": cfg | {"tier_mix": tier_mix,
                                "features": feat_order,
                                "n_train": int(len(ytr)),
                                "n_test": int(len(yte)),
                                "label_prevalence": float(y.mean()),
                                "tier_counts": tier_counts},
        "metrics": {k: {kk: vv for kk, vv in v.items() if kk != "history"}
                    for k, v in results.items()},
        "costs": {k: vars(v) for k, v in costs.items()},
        "comms": comms,
        "contribution": contrib,
        "sweep": {"n_homes": sweep_homes, **sweep},
        "histories": histories,
        "platform": {"machine": platform.machine(), "system": platform.system(),
                     "processor": platform.processor(),
                     "energy_source": costs["tieraware_fl"].energy_source},
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2))

    # tidy CSV of the headline comparison
    rows = []
    for k in ["centralized", "intersection_fl", "naive_union_fl", "tieraware_fl"]:
        rows.append({"method": k, **results[k], **vars(costs[k])})
    pd.DataFrame(rows).drop(columns=["history"]).to_csv(
        out_dir / "comparison.csv", index=False)

    _print_summary(results, costs, comms, contrib)
    make_figures(results, histories, sweep, sweep_homes, contrib, costs, fig_dir)
    print(f"\nWrote {out_dir}/results.json, comparison.csv and {fig_dir}/*.png")
    return 0


def _print_summary(results, costs, comms, contrib):
    print("\n=== Headline (test-set AUROC) ===")
    for k in ["centralized", "intersection_fl", "naive_union_fl", "tieraware_fl"]:
        r = results[k]
        print(f"  {k:18s} AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
              f"Brier={r['brier']:.4f}")
    gap_int = results["centralized"]["auroc"] - results["intersection_fl"]["auroc"]
    gap_ta = results["centralized"]["auroc"] - results["tieraware_fl"]["auroc"]
    recov = (results["tieraware_fl"]["auroc"] - results["intersection_fl"]["auroc"])
    print(f"  -> intersection gap vs centralized : {gap_int*100:.2f} pp")
    print(f"  -> tier-aware  gap vs centralized  : {gap_ta*100:.2f} pp")
    print(f"  -> tier-aware recovers over intersection: +{recov*100:.2f} pp AUROC")
    print("\n=== On-device cost (tier-aware FL) ===")
    c = costs["tieraware_fl"]
    print(f"  wall={c.wall_s:.2f}s cpu={c.cpu_s:.2f}s energy={c.energy_j:.1f}J "
          f"({c.energy_source})")
    print(f"  uplink/home/round={comms['bytes_per_home_per_round']} B  "
          f"total={comms['total_bytes_tieraware']/1024:.1f} KiB  "
          f"raw egress={comms['raw_data_egress_bytes']} B")
    print("\n=== Tier contribution / reward ladder ===")
    print(f"  profile-only AUROC={contrib['profile_auc']:.4f}")
    for t in contrib["tiers"]:
        print(f"  Tier {t['tier']} ({t['name']:18s}) cumAUROC={t['cum_auc']:.4f} "
              f"Δ={t['marginal_auc_gain']*100:+.2f}pp  "
              f"learned_reward={t['learned_reward_share']:.2f}  "
              f"product_credit={t['product_credit']:.2f}")


def make_figures(results, histories, sweep, sweep_homes, contrib, costs, fig_dir):
    plt.rcParams.update({"font.size": 11, "figure.dpi": 150})
    styles = {"intersection_fl": ("#9e9e9e", "-"),
              "naive_union_fl": ("#e0884e", "-"),
              "tieraware_fl": ("#2e8b57", "-")}

    # Fig 1: headline AUROC bars
    order = ["centralized", "intersection_fl", "naive_union_fl", "tieraware_fl"]
    aucs = [results[k]["auroc"] for k in order]
    colors = ["#444", "#9e9e9e", "#e0884e", "#2e8b57"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar([PRETTY[k] for k in order], aucs, color=colors)
    for b, a in zip(bars, aucs, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, a + 0.003, f"{a:.3f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Test AUROC")
    ax.set_ylim(min(aucs) - 0.03, max(aucs) + 0.03)
    ax.set_title("Diabetes risk scoring under device-tier heterogeneity")
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_auroc.png"); plt.close(fig)

    # Fig 2: convergence
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    cen = results["centralized"]["auroc"]
    ax.axhline(cen, ls="--", c="#444", label="Centralized (upper bound)")
    for m, hist in histories.items():
        c, ls = styles[m]
        ax.plot(range(1, len(hist) + 1), hist, ls, color=c,
                label=PRETTY[m].replace("\n", " "))
    ax.set_xlabel("Federated round"); ax.set_ylabel("Test AUROC")
    ax.set_title("Convergence"); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_convergence.png"); plt.close(fig)

    # Fig 3: tier contribution vs product credit ladder
    tiers = contrib["tiers"]
    labels = [f"T{t['tier']}\n{t['name']}" for t in tiers]
    learned = [t["learned_reward_share"] for t in tiers]
    credit = [t["product_credit"] for t in tiers]
    x = np.arange(len(tiers)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - w / 2, learned, w, label="Learned reward share", color="#2e8b57")
    ax2 = ax.twinx()
    ax2.bar(x + w / 2, credit, w, label="Product premium credit", color="#7aa6c2")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Learned reward share"); ax2.set_ylabel("Product premium credit")
    ax.set_title("Tier contribution vs. premium-credit ladder")
    l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la1 + la2, fontsize=8.5, loc="upper left")
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_contribution.png"); plt.close(fig)

    # Fig 4: scaling sweep
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in FL_METHODS:
        c, ls = styles[m]
        ax.plot(sweep_homes, sweep[m], "o-", color=c,
                label=PRETTY[m].replace("\n", " "))
    ax.axhline(results["centralized"]["auroc"], ls="--", c="#444",
               label="Centralized")
    ax.set_xlabel("Number of edge nodes (homes)"); ax.set_ylabel("Test AUROC")
    ax.set_title("Scaling across federated nodes"); ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_scaling.png"); plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
