"""The device -> feature -> tier mapping is the conceptual core; pin its invariants."""
import config as C


def test_profile_features_need_no_device():
    # Entered once at app setup, so every tier observes them.
    for tier in C.TIER_FEATURES:
        assert set(C.PROFILE) <= set(C.features_for_tier(tier))


def test_tiers_are_strictly_cumulative():
    prev = set(C.features_for_tier(1))
    for tier in sorted(C.TIER_FEATURES)[1:]:
        cur = set(C.features_for_tier(tier))
        assert prev < cur, f"tier {tier} must strictly add features to tier {tier-1}"
        assert cur - prev == set(C.TIER_FEATURES[tier])
        prev = cur


def test_no_feature_is_contributed_by_two_tiers():
    seen = set(C.PROFILE)
    for tier in sorted(C.TIER_FEATURES):
        feats = set(C.TIER_FEATURES[tier])
        assert not (feats & seen), f"tier {tier} re-declares {feats & seen}"
        seen |= feats


def test_all_features_is_the_top_tier_and_has_no_duplicates():
    top = max(C.TIER_FEATURES)
    assert C.all_features() == C.features_for_tier(top)
    assert len(C.all_features()) == len(set(C.all_features()))


def test_feature_order_is_deterministic():
    # results.json records the column order; a reshuffle would silently invalidate it.
    assert C.all_features() == ["age", "sex_male", "current_smoker", "bmi",
                                "weight", "sbp", "dbp", "pulse", "waist", "hba1c"]


def test_product_credit_ladder_is_monotone():
    credits = [C.TIER_CREDIT[t] for t in sorted(C.TIER_CREDIT)]
    assert credits == sorted(credits)
    assert set(C.TIER_CREDIT) == set(C.TIER_FEATURES) == set(C.TIER_NAMES)


def test_label_is_not_a_feature():
    # The label must never leak into the feature matrix.
    assert C.LABEL not in C.all_features()


def test_medication_count_is_not_used():
    # Deliberately excluded: diagnosis-adjacent for a prevalence label (see README).
    joined = " ".join(C.all_features())
    for banned in ("med", "rx", "drug", "pill"):
        assert banned not in joined
