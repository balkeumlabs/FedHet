"""Guard the published artifact: results.json must stay internally consistent, and
the README's headline table must keep matching it.

These tests read only files committed to the repository -- they do not retrain
anything -- so they run in CI without network access and catch the most common
publication failure: a number in the prose drifting away from the number in the
results file.
"""
import json
import re

import pytest

from conftest import RESULTS_JSON, README
from verify_results import rounds_to_converge, CONVERGENCE_TOL

METHODS = ["centralized", "intersection_fl", "naive_union_fl", "tieraware_fl"]
FL_METHODS = METHODS[1:]

# README table row label -> results.json key.
ROW_TO_METHOD = {
    "Centralized (upper bound)": "centralized",
    "Intersection FL (Tier-1 only)": "intersection_fl",
    "Naive union FL": "naive_union_fl",
    "Tier-aware FL (ours)": "tieraware_fl",
}


@pytest.fixture(scope="module")
def res():
    return json.loads(RESULTS_JSON.read_text())


@pytest.fixture(scope="module")
def readme():
    return README.read_text()


def _clean(cell: str) -> str:
    return cell.replace("**", "").replace("−", "-").strip()


@pytest.fixture(scope="module")
def readme_table(readme):
    """Parse the TL;DR results table into {method_key: [cells]}."""
    rows = {}
    for line in readme.splitlines():
        if not line.startswith("|"):
            continue
        cells = [_clean(c) for c in line.strip().strip("|").split("|")]
        if cells and cells[0] in ROW_TO_METHOD:
            rows[ROW_TO_METHOD[cells[0]]] = cells
    return rows


# ----- internal consistency -------------------------------------------------
def test_results_json_has_every_method(res):
    assert set(res["metrics"]) == set(METHODS)
    assert set(res["histories"]) == set(FL_METHODS)


def test_every_history_has_one_entry_per_configured_round(res):
    for m in FL_METHODS:
        assert len(res["histories"][m]) == res["config"]["rounds"]


def test_reported_metrics_are_in_range(res):
    for m in METHODS:
        assert 0.5 <= res["metrics"][m]["auroc"] <= 1.0
        assert 0.0 <= res["metrics"][m]["auprc"] <= 1.0
        assert 0.0 <= res["metrics"][m]["brier"] <= 0.25


def test_centralized_is_the_upper_bound(res):
    cen = res["metrics"]["centralized"]["auroc"]
    for m in FL_METHODS:
        assert res["metrics"][m]["auroc"] <= cen


def test_tier_aware_beats_both_federated_baselines(res):
    ta = res["metrics"]["tieraware_fl"]["auroc"]
    assert ta > res["metrics"]["naive_union_fl"]["auroc"]
    assert ta > res["metrics"]["intersection_fl"]["auroc"]


def test_final_history_entry_matches_the_reported_auroc(res):
    for m in FL_METHODS:
        assert res["histories"][m][-1] == pytest.approx(
            res["metrics"][m]["auroc"], abs=1e-9)


def test_feature_count_matches_the_communication_accounting(res):
    d = len(res["config"]["features"])
    assert res["comms"]["model_params"] == d + 1
    assert res["comms"]["bytes_per_home_per_round"] == (d + 1) * 4
    assert res["comms"]["raw_data_egress_bytes"] == 0


def test_total_uplink_is_consistent_with_rounds_and_homes(res):
    per_round = res["comms"]["bytes_per_round_all_homes"]
    assert res["comms"]["total_bytes_tieraware"] == per_round * res["config"]["rounds"]
    n_homes = sum(res["config"]["tier_counts"].values())
    assert per_round == res["comms"]["bytes_per_home_per_round"] * n_homes


def test_cohort_split_adds_up(res):
    cfg = res["config"]
    n = cfg["n_train"] + cfg["n_test"]
    assert cfg["n_test"] == pytest.approx(0.2 * n, abs=1)
    assert 0.10 < cfg["label_prevalence"] < 0.20


def test_tier_contribution_ladder_is_monotone(res):
    tiers = res["contribution"]["tiers"]
    assert [t["tier"] for t in tiers] == [1, 2, 3]
    cum = [t["cum_auc"] for t in tiers]
    assert cum == sorted(cum), "cumulative AUROC must not fall as devices are added"
    shares = [t["learned_reward_share"] for t in tiers]
    assert shares == sorted(shares)
    assert shares[-1] == pytest.approx(1.0)


def test_top_tier_cumulative_auroc_equals_the_centralized_model(res):
    # Tier 3 observes every feature, so its cumulative fit IS the centralized fit.
    assert res["contribution"]["tiers"][-1]["cum_auc"] == pytest.approx(
        res["metrics"]["centralized"]["auroc"], abs=1e-9)


def test_marginal_gains_reconstruct_the_cumulative_curve(res):
    base = res["contribution"]["profile_auc"]
    acc = base
    for t in res["contribution"]["tiers"]:
        acc += t["marginal_auc_gain"]
        assert acc == pytest.approx(t["cum_auc"], abs=1e-9)


def test_published_run_used_measured_rapl_energy(res):
    """The energy figures in the paper are measured, not TDP estimates."""
    assert res["platform"]["energy_source"] == "rapl"
    for m in METHODS:
        assert res["costs"][m]["energy_source"] == "rapl"
        assert res["costs"][m]["energy_j"] > 0.0


def test_no_stale_cvd_naming_survives(res):
    """The study switched from a CVD label to diabetes; nothing may still say CVD."""
    assert "cvd" not in json.dumps(res).lower()


# ----- README <-> results.json ----------------------------------------------
def test_readme_table_covers_every_method(readme_table):
    assert set(readme_table) == set(METHODS)


def test_readme_auroc_column_matches_results(res, readme_table):
    for method, cells in readme_table.items():
        assert float(cells[1]) == pytest.approx(
            res["metrics"][method]["auroc"], abs=5e-4), f"README AUROC for {method}"


def test_readme_auprc_column_matches_results(res, readme_table):
    for method, cells in readme_table.items():
        assert float(cells[2]) == pytest.approx(
            res["metrics"][method]["auprc"], abs=5e-4), f"README AUPRC for {method}"


def test_readme_gap_column_matches_results(res, readme_table):
    cen = res["metrics"]["centralized"]["auroc"]
    for method, cells in readme_table.items():
        if method == "centralized":
            continue
        want = (res["metrics"][method]["auroc"] - cen) * 100
        got = float(cells[3].replace("pp", "").strip())
        assert got == pytest.approx(want, abs=0.05), f"README gap for {method}"


def test_readme_rounds_to_converge_column_matches_histories(res, readme_table):
    for method, cells in readme_table.items():
        if method == "centralized":
            continue
        got = int(re.match(r"\d+", cells[4]).group())
        assert got == rounds_to_converge(res["histories"][method]), \
            f"README rounds-to-converge for {method} (tol {CONVERGENCE_TOL})"


def test_readme_headline_deltas_match_results(res, readme):
    cen = res["metrics"]["centralized"]["auroc"]
    inter = res["metrics"]["intersection_fl"]["auroc"]
    ta = res["metrics"]["tieraware_fl"]["auroc"]
    for claim, value in (("15.6 AUROC points", (cen - inter) * 100),
                         ("+13.9 points", (ta - inter) * 100),
                         ("within 1.6 points", (cen - ta) * 100)):
        assert claim in readme, f"README no longer states {claim!r}"
        stated = float(re.search(r"[\d.]+", claim).group())
        assert stated == pytest.approx(value, abs=0.05)


def test_readme_energy_and_uplink_claims_match_results(res, readme):
    ta = res["costs"]["tieraware_fl"]
    assert f"{ta['energy_j']:.1f} J" in readme.replace("**", "")
    assert f"{res['comms']['bytes_per_home_per_round']} bytes" in readme.replace("**", "")


def test_readme_tier_contribution_claims_match_results(res, readme):
    flat = readme.replace("**", "")
    for t in res["contribution"]["tiers"]:
        assert f"+{t['marginal_auc_gain']*100:.1f} pp" in flat, \
            f"README tier-{t['tier']} marginal gain"
