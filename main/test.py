"""Tests for the properties the paper actually claims.

This is not coverage for its own sake. Every claim in the write-up that a
reader cannot check by eye is a property of the code in `src/`, and each one is
asserted here:

  * blocked features are really blocked (Section IV-A, the leakage audit)
  * a mother never crosses the train/test boundary (Section IV-C)
  * the operating point never exceeds the flag-rate cap (Section IV-D)
  * ensemble weights are counts over a fixed budget (Table III)
  * the calibration guard rejects a model that has shrunk past the base rate
  * conformal sets follow the quantile rule

Nothing here touches the BDHS data, which is not redistributable. Every test
builds its own arrays, so this runs on a clean checkout with no download.

    pip install pytest
    python test.py            # or: pytest test.py -q

Note the explicit filename. pytest discovers `test_*.py` and `*_test.py`, and
a bare `test.py` matches neither, so `pytest` with no arguments will report
that it collected nothing. The __main__ block at the foot of the file exists so
that `python test.py` works regardless.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# This file sits beside train.py and src/, so its own directory is the import
# root. Explicit rather than relying on pytest's rootdir insertion, which
# depends on how the file is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation import choose_threshold, net_benefit           # noqa: E402
from src.feature_engineering import (FERTILITY_ACCOUNTING,         # noqa: E402
                                     Z_SCORES, blacklist_for,
                                     features_for)
from src.models import (ENSEMBLE_SIZE, GreedyEnsemble,             # noqa: E402
                        calibration_metrics, greedy_select,
                        prediction_set)
from src.preprocessing import grouped_split                        # noqa: E402
from src.utils import MAX_FLAG_RATE, denullify, fn_cost, get_scores  # noqa: E402

SEED = 0


# ===========================================================================
# 1. Leakage. The audit is the paper's central methodological claim, and it
#    is enforced entirely by these two functions.
# ===========================================================================
class TestLeakageBlocking:

    def test_fertility_accounting_blocked_for_neonatal_death(self):
        """children-ever-born minus living children IS the death count.

        Leaving it in took neonatal ROC-AUC from 0.773 to 0.967, so this is
        the single most important assertion in the file.
        """
        blocked = set(blacklist_for("neonatal_death"))
        assert set(FERTILITY_ACCOUNTING) <= blocked
        assert "b12" in blocked, "interval to the NEXT birth is post-outcome"

    def test_z_scores_blocked_for_every_nutrition_target(self):
        """haz below -2 IS stunting; the target would be its own feature."""
        for target in ("stunted", "severe_stunting", "underweight_child"):
            assert set(Z_SCORES) <= set(blacklist_for(target)), target

    def test_nutrition_blocks_do_not_leak_into_other_targets(self):
        """A blanket blacklist would silently weaken unrelated targets."""
        assert not set(Z_SCORES) & set(blacklist_for("facility_delivery"))
        assert not set(FERTILITY_ACCOUNTING) & set(blacklist_for("stunted"))

    def test_registry_blacklist_is_unioned_not_replaced(self):
        registry = {"neonatal_death": {"blacklist": ["custom_blocked_col"]}}
        blocked = set(blacklist_for("neonatal_death", registry))
        assert "custom_blocked_col" in blocked
        assert set(FERTILITY_ACCOUNTING) <= blocked, "built-ins still apply"

    def test_features_for_drops_blocked_columns(self):
        """The regression that matters: a blocked name listed as a feature.

        The registry is written by an earlier stage, so it can and does list
        columns that the blacklist blocks. features_for is the last gate.
        """
        registry = {"neonatal_death": {
            "features": ["v012", "v201", "v106", "b12", "v190"],
            "blacklist": [],
        }}
        available = ["v012", "v201", "v106", "b12", "v190"]
        feats = features_for("neonatal_death", registry, available)
        assert feats == ["v012", "v106", "v190"]
        assert "v201" not in feats and "b12" not in feats

    def test_features_for_drops_columns_absent_from_the_cohort(self):
        """A feature valid in one DHS universe may not exist in another."""
        registry = {"stunted": {"features": ["v012", "m14", "v190"],
                                "blacklist": []}}
        feats = features_for("stunted", registry, ["v012", "v190"])
        assert feats == ["v012", "v190"]

    def test_resolved_features_are_disjoint_from_the_blacklist(self):
        """The invariant, stated directly, over every shipped target."""
        registry = {
            "neonatal_death": {"features": FERTILITY_ACCOUNTING + ["v012"],
                               "blacklist": []},
            "stunted": {"features": Z_SCORES + ["v012"], "blacklist": []},
        }
        for target, spec in registry.items():
            feats = features_for(target, registry, spec["features"])
            assert not set(feats) & set(blacklist_for(target, registry))


# ===========================================================================
# 2. Grouped splitting. 90% of birth rows share a mother with another row, so
#    a random split would put siblings on both sides.
# ===========================================================================
class TestGroupedSplit:

    @staticmethod
    def _cohort(n_mothers=200, births_per_mother=3):
        rng = np.random.default_rng(SEED)
        mothers = np.repeat(np.arange(n_mothers), births_per_mother)
        n = len(mothers)
        X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
        y = pd.Series(rng.integers(0, 2, size=n))
        return X, y, pd.Series(mothers)

    def test_no_mother_appears_on_both_sides(self):
        X, y, g = self._cohort()
        _, _, _, _, g_tr, g_te = grouped_split(X, y, g)
        assert not set(g_tr) & set(g_te)

    def test_every_row_is_used_exactly_once(self):
        X, y, g = self._cohort()
        X_tr, X_te, *_ = grouped_split(X, y, g)
        assert len(X_tr) + len(X_te) == len(X)
        assert not set(X_tr.index) & set(X_te.index)

    def test_split_is_reproducible_for_a_fixed_seed(self):
        X, y, g = self._cohort()
        a = grouped_split(X, y, g, random_state=SEED)
        b = grouped_split(X, y, g, random_state=SEED)
        assert list(a[1].index) == list(b[1].index)

    def test_a_random_split_would_have_leaked(self):
        """Shows the guard is load-bearing, not decorative."""
        X, y, g = self._cohort()
        rng = np.random.default_rng(SEED)
        shuffled = rng.permutation(len(X))
        cut = int(0.8 * len(X))
        naive_tr, naive_te = g.iloc[shuffled[:cut]], g.iloc[shuffled[cut:]]
        assert set(naive_tr) & set(naive_te), (
            "if this ever passes the fixture stopped having repeated mothers")


# ===========================================================================
# 3. Operating points. A cost function alone drove stunting to flag 60% of
#    children, which no service can act on.
# ===========================================================================
class TestThresholdPolicy:

    @staticmethod
    def _scored(n=2000, prevalence=0.20):
        rng = np.random.default_rng(SEED)
        y = (rng.random(n) < prevalence).astype(int)
        scores = np.clip(rng.normal(0.35 + 0.30 * y, 0.18), 0.01, 0.99)
        return y, scores

    def test_flag_rate_respects_the_cap(self):
        y, s = self._scored()
        _, flag_rate, _ = choose_threshold(y, s, fn_cost=10)
        assert flag_rate <= MAX_FLAG_RATE + 1e-9

    def test_cap_binds_even_at_a_punitive_false_negative_cost(self):
        """Without the cap a 50x cost flags nearly everyone."""
        y, s = self._scored()
        _, flag_rate, _ = choose_threshold(y, s, fn_cost=50)
        assert flag_rate <= MAX_FLAG_RATE + 1e-9

        _, uncapped, _ = choose_threshold(y, s, fn_cost=50, max_flag_rate=1.0)
        assert uncapped > MAX_FLAG_RATE

    def test_falls_back_to_min_cost_when_the_cap_is_unreachable(self):
        y, s = self._scored()
        t, flag_rate, curve = choose_threshold(y, s, fn_cost=5,
                                               max_flag_rate=0.0)
        assert np.isfinite(t)
        assert flag_rate == pytest.approx(
            curve.loc[curve["cost"].idxmin(), "flag_rate"])

    def test_a_higher_miss_cost_never_raises_the_threshold(self):
        y, s = self._scored()
        cheap, _, _ = choose_threshold(y, s, fn_cost=1, max_flag_rate=1.0)
        dear, _, _ = choose_threshold(y, s, fn_cost=20, max_flag_rate=1.0)
        assert dear <= cheap

    def test_the_returned_threshold_reproduces_the_returned_flag_rate(self):
        y, s = self._scored()
        t, flag_rate, _ = choose_threshold(y, s, fn_cost=5)
        assert (s >= t).mean() == pytest.approx(flag_rate)

    def test_cost_policy_matches_the_paper(self):
        assert fn_cost("neonatal_death", "adverse") == 10
        assert fn_cost("stunted", "adverse") == 5
        assert fn_cost("facility_delivery", "pathway") == 1


# ===========================================================================
# 4. Net benefit. Bounded above by prevalence, which is the point of reporting
#    it for a 4% outcome.
# ===========================================================================
class TestNetBenefit:

    def test_treat_none_scores_zero(self):
        y = np.array([0, 1, 0, 1, 0])
        p = np.zeros(5)
        assert net_benefit(y, p, [0.2]) == pytest.approx([0.0])

    def test_a_perfect_model_reaches_prevalence(self):
        y = np.array([0, 1, 0, 1, 0, 0, 1, 0])
        nb = net_benefit(y, y.astype(float), [0.3])[0]
        assert nb == pytest.approx(y.mean())

    def test_false_positives_are_penalised_by_the_odds_of_the_threshold(self):
        y = np.array([0, 0, 0, 0])
        nb = net_benefit(y, np.ones(4), [0.5])[0]
        assert nb == pytest.approx(-1.0)      # 0 - 1 * (0.5/0.5)


# ===========================================================================
# 5. The ensemble. Table III states the weights are counts summing to fifteen.
# ===========================================================================
class TestGreedyEnsemble:

    @staticmethod
    def _preds(n=600):
        rng = np.random.default_rng(SEED)
        y = (rng.random(n) < 0.3).astype(int)
        return y, {
            "good": np.clip(0.5 * y + rng.normal(0.25, 0.10, n), 0, 1),
            "ok": np.clip(0.2 * y + rng.normal(0.35, 0.15, n), 0, 1),
            "noise": rng.random(n),
        }

    def test_counts_sum_to_the_budget(self):
        y, preds = self._preds()
        counts, history = greedy_select(preds, y)
        assert sum(counts.values()) == ENSEMBLE_SIZE
        assert len(history) == ENSEMBLE_SIZE

    def test_selection_is_with_replacement(self):
        y, preds = self._preds()
        counts, _ = greedy_select(preds, y)
        assert max(counts.values()) > 1, "fifteen rounds over three learners"

    def test_the_informative_learner_is_preferred_to_noise(self):
        y, preds = self._preds()
        counts, _ = greedy_select(preds, y)
        assert counts["good"] > counts["noise"]

    def test_weights_normalise_and_drop_unselected_members(self):
        ens = GreedyEnsemble(members={"a": None, "b": None, "c": None},
                             counts={"a": 9, "b": 6, "c": 0}).fit()
        assert sum(ens.weights.values()) == pytest.approx(1.0)
        assert ens.weights["a"] == pytest.approx(9 / 15)
        assert "c" not in ens.weights, "a zero-weight member must not vote"

    def test_prediction_is_the_weighted_average_of_members(self):
        class Stub:
            def __init__(self, value):
                self.value = value

            def predict_proba(self, X):
                p = np.full(len(X), self.value)
                return np.column_stack([1 - p, p])

        ens = GreedyEnsemble(members={"a": Stub(1.0), "b": Stub(0.0)},
                             counts={"a": 10, "b": 5}).fit()
        X = pd.DataFrame({"x": range(4)})
        assert ens.predict_proba(X)[:, 1] == pytest.approx(10 / 15)
        assert list(ens.predict(X)) == [1, 1, 1, 1]


# ===========================================================================
# 6. Calibration. The two-criterion guard exists because Brier alone accepted
#    a model that had learned to predict near-zero for everyone.
# ===========================================================================
class TestCalibration:

    @staticmethod
    def _calibrated(n=4000):
        rng = np.random.default_rng(SEED)
        p = rng.uniform(0.05, 0.95, n)
        return (rng.random(n) < p).astype(int), p

    def test_a_calibrated_model_scores_slope_one_and_citl_zero(self):
        y, p = self._calibrated()
        slope, citl = calibration_metrics(y, p)
        assert slope == pytest.approx(1.0, abs=0.15)
        assert citl == pytest.approx(0.0, abs=0.15)

    def test_overconfident_probabilities_give_a_negative_citl(self):
        """Predicting too high must push the intercept below zero."""
        y, p = self._calibrated()
        _, citl = calibration_metrics(y, np.clip(p * 2.5, 1e-6, 1 - 1e-6))
        assert citl < -0.3


class TestCalibrationGuard:
    """The two-criterion guard, exercised through keep_calibrator itself.

    Reproduces the severe_stunting case from the comment in models.py: a
    calibrator that improved Brier from 0.0844 to 0.0753 by learning to
    predict close to zero for everyone, which on a 7.5% outcome scores well
    and is useless.
    """

    class _Stub:
        def __init__(self, p):
            self.p = p

        def predict_proba(self, X):
            p = np.asarray(self.p, dtype=float)
            if p.ndim == 0:
                p = np.full(len(X), float(p))
            return np.column_stack([1 - p, p])

    @staticmethod
    def _dataset(y):
        return {"y_te": pd.Series(y),
                "X_te": pd.DataFrame({"x": np.arange(len(y))})}

    @staticmethod
    def _rare_outcome(n=4000, prevalence=0.075):
        rng = np.random.default_rng(SEED)
        y = (rng.random(n) < prevalence).astype(int)
        inflated = np.clip(rng.normal(0.25, 0.08, n), 0.01, 0.99)
        return y, inflated

    def test_brier_alone_would_have_accepted_the_degenerate_calibrator(self):
        """Establishes the premise: this is a Brier improvement."""
        from sklearn.metrics import brier_score_loss
        y, raw = self._rare_outcome()
        assert brier_score_loss(y, np.full(len(y), 0.002)) < \
            brier_score_loss(y, raw)

    def test_a_calibrator_that_shrinks_past_the_base_rate_is_rejected(self):
        from src.models import keep_calibrator
        y, raw = self._rare_outcome()
        keep, reason = keep_calibrator(self._Stub(raw), self._Stub(0.002),
                                       self._dataset(y))
        assert keep is False
        assert "CITL" in reason, f"rejected for the wrong reason: {reason}"

    def test_a_calibrator_that_improves_both_criteria_is_kept(self):
        """Positive control -- the guard must not reject everything."""
        from src.models import keep_calibrator
        y, raw = self._rare_outcome()
        keep, _ = keep_calibrator(self._Stub(raw),
                                  self._Stub(float(y.mean())),
                                  self._dataset(y))
        assert keep is True

    def test_a_calibrator_that_worsens_brier_is_rejected_on_brier(self):
        from src.models import keep_calibrator
        y, raw = self._rare_outcome()
        keep, reason = keep_calibrator(self._Stub(float(y.mean())),
                                       self._Stub(raw), self._dataset(y))
        assert keep is False
        assert "Brier" in reason


# ===========================================================================
# 7. Conformal sets.
# ===========================================================================
class TestPredictionSet:

    def test_a_confident_positive_returns_only_the_positive_label(self):
        assert prediction_set(0.95, quantile=0.20) == [1]

    def test_a_confident_negative_returns_only_the_negative_label(self):
        assert prediction_set(0.05, quantile=0.20) == [0]

    def test_an_uncertain_score_abstains_with_both_labels(self):
        assert sorted(prediction_set(0.50, quantile=0.60)) == [0, 1]

    def test_a_tight_quantile_can_return_the_empty_set(self):
        assert prediction_set(0.50, quantile=0.10) == []

    @pytest.mark.parametrize("p", [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0])
    def test_membership_follows_the_quantile_rule(self, p):
        q = 0.30
        s = prediction_set(p, q)
        assert (1 in s) == ((1 - p) <= q)
        assert (0 in s) == (p <= q)


# ===========================================================================
# 8. Frame handling. denullify exists because sklearn raises "boolean value of
#    NA is ambiguous" on every estimator otherwise.
# ===========================================================================
class TestDenullify:

    def test_no_pandas_na_survives(self):
        df = pd.DataFrame({
            "i": pd.array([1, None, 3], dtype="Int64"),
            "f": pd.array([1.5, None, 3.5], dtype="Float64"),
            "b": pd.array([True, None, False], dtype="boolean"),
            "s": pd.array(["a", None, "c"], dtype="string"),
            "c": pd.Categorical(["x", None, "z"]),
        })
        out = denullify(df)
        for col in out.columns:
            assert not out[col].isin([pd.NA]).any(), col

    def test_numeric_columns_become_numpy_floats_with_nan(self):
        df = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int64")})
        out = denullify(df)
        assert out["i"].dtype == np.float64
        assert np.isnan(out["i"].iloc[1])

    def test_values_are_preserved(self):
        df = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int64"),
                           "s": pd.array(["a", None, "c"], dtype="string")})
        out = denullify(df)
        assert out["i"].iloc[0] == 1.0 and out["i"].iloc[2] == 3.0
        assert out["s"].iloc[0] == "a" and out["s"].iloc[2] == "c"

    def test_the_input_frame_is_not_mutated(self):
        df = pd.DataFrame({"i": pd.array([1, None, 3], dtype="Int64")})
        denullify(df)
        assert str(df["i"].dtype) == "Int64"


# ===========================================================================
# 9. Score extraction. Half the panel has no predict_proba.
# ===========================================================================
class TestGetScores:

    def test_predict_proba_is_used_when_available(self):
        class WithProba:
            def predict_proba(self, X):
                p = np.linspace(0.1, 0.9, len(X))
                return np.column_stack([1 - p, p])

        X = pd.DataFrame({"x": range(5)})
        assert get_scores(WithProba(), X) == pytest.approx(
            np.linspace(0.1, 0.9, 5))

    def test_decision_function_is_min_max_scaled_into_the_unit_interval(self):
        class WithMargin:
            def decision_function(self, X):
                return np.array([-2.0, 0.0, 3.0])

        s = get_scores(WithMargin(), pd.DataFrame({"x": range(3)}))
        assert s.min() == pytest.approx(0.0)
        assert s.max() == pytest.approx(1.0)
        assert np.all(np.diff(s) > 0), "scaling must preserve the ranking"

    def test_a_constant_margin_does_not_divide_by_zero(self):
        class Flat:
            def decision_function(self, X):
                return np.zeros(4)

        s = get_scores(Flat(), pd.DataFrame({"x": range(4)}))
        assert np.all(np.isfinite(s))


# ===========================================================================
# Entry point. `pytest` with no arguments will not collect this file -- its
# default patterns are test_*.py and *_test.py -- so running it directly hands
# pytest the path explicitly.
# ===========================================================================
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
