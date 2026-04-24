"""Statistical analysis for EXP_010 (accuracy vs L0) and EXP_015 (t-tests, Cohen's d).

EXP_010: Tests whether the L0 reduction at alpha=1.0 is explained by accuracy alone
         (the "weaker model" confound). Fits L0 ~ val_acc linear regression and checks
         whether topo_strong residuals are significantly negative.

EXP_015: Paired t-tests and Cohen's d for H2 (L0, dead%) and H3 (patch_ratio) across
         seeds. Paired by seed to control for random variation in initialisation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy import stats as scipy_stats


# ── Data loading ─────────────────────────────────────────────────────────────

def load_results(json_dir: Path) -> list[dict]:
    """Load all *_analysis.json files. Returns only multi-seed ViT-S/16 runs."""
    records = []
    for p in sorted(json_dir.glob("*_analysis.json")):
        r = json.loads(p.read_text())
        # Skip runs without a seed tag (early TinyViT / imagenet100 single runs)
        if "_s" not in p.stem:
            continue
        records.append(r)
    return records


def filter_main(records: list[dict], alphas=(0.0, 0.1, 1.0), seeds=(42, 123, 456)) -> list[dict]:
    return [r for r in records if r["alpha"] in alphas and r["seed"] in seeds]


# ── EXP_010: Accuracy vs L0 regression ──────────────────────────────────────

class RegressionResult(NamedTuple):
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    residuals: dict[str, float]  # run_id -> residual


def accuracy_vs_l0_regression(records: list[dict]) -> RegressionResult:
    """Fit L0 ~ val_acc, return regression and per-run residuals."""
    run_ids = [r["run_id"] for r in records]
    accs = np.array([r["val_acc"] for r in records])
    l0s = np.array([r["h2"]["l0_norm"] for r in records])

    slope, intercept, r, p, _ = scipy_stats.linregress(accs, l0s)
    predicted = slope * accs + intercept
    residuals = {rid: float(res) for rid, res in zip(run_ids, l0s - predicted)}

    return RegressionResult(
        slope=float(slope),
        intercept=float(intercept),
        r_squared=float(r**2),
        p_value=float(p),
        residuals=residuals,
    )


# ── EXP_015: Paired t-tests + Cohen's d ─────────────────────────────────────

class PairedTestResult(NamedTuple):
    metric: str
    comparison: str        # e.g. "alpha=0.0 vs alpha=1.0"
    baseline_values: list[float]
    treatment_values: list[float]
    mean_diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    ci_low: float          # 95% bootstrap CI on mean difference
    ci_high: float


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = b - a
    return float(diff.mean() / (diff.std(ddof=1) + 1e-12))


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    diffs = b - a
    boot = rng.choice(diffs, size=(n_boot, len(diffs)), replace=True).mean(axis=1)
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return lo, hi


METRICS = {
    "h2.l0_norm":     ("h2", "l0_norm",     "H2: L0 Norm",     "lower is better"),
    "h2.dead_pct":    ("h2", "dead_pct",     "H2: Dead %",      "higher means more wasted capacity"),
    "h3.patch_ratio": ("h3", "patch_ratio",  "H3: Patch Ratio", "higher is more causally pure"),
}


def paired_tests(
    records: list[dict],
    baseline_alpha: float = 0.0,
    treatment_alphas: tuple[float, ...] = (0.1, 1.0),
    seeds: tuple[int, ...] = (42, 123, 456),
    n_boot: int = 10_000,
) -> list[PairedTestResult]:
    """Run paired t-tests for each metric and each baseline-vs-treatment pair."""

    def get_values(alpha: float, metric_key: str) -> np.ndarray:
        section, field, _, _ = METRICS[metric_key]
        vals = []
        for seed in seeds:
            match = [r for r in records if r["alpha"] == alpha and r["seed"] == seed]
            if not match:
                raise ValueError(f"Missing run: alpha={alpha}, seed={seed}")
            vals.append(match[0][section][field])
        return np.array(vals)

    results = []
    for metric_key in METRICS:
        _, _, label, _ = METRICS[metric_key]
        base_vals = get_values(baseline_alpha, metric_key)
        for treat_alpha in treatment_alphas:
            treat_vals = get_values(treat_alpha, metric_key)
            t_stat, p_val = scipy_stats.ttest_rel(base_vals, treat_vals)
            d = _cohens_d(base_vals, treat_vals)
            lo, hi = _bootstrap_ci(base_vals, treat_vals, n_boot=n_boot)
            results.append(PairedTestResult(
                metric=label,
                comparison=f"α={baseline_alpha} vs α={treat_alpha}",
                baseline_values=base_vals.tolist(),
                treatment_values=treat_vals.tolist(),
                mean_diff=float((treat_vals - base_vals).mean()),
                t_stat=float(t_stat),
                p_value=float(p_val),
                cohens_d=d,
                ci_low=lo,
                ci_high=hi,
            ))
    return results


# ── Summary helpers ───────────────────────────────────────────────────────────

def results_to_dict(reg: RegressionResult, tests: list[PairedTestResult]) -> dict:
    return {
        "exp010_accuracy_l0_regression": {
            "slope": reg.slope,
            "intercept": reg.intercept,
            "r_squared": reg.r_squared,
            "p_value": reg.p_value,
            "residuals": reg.residuals,
        },
        "exp015_paired_tests": [
            {
                "metric": t.metric,
                "comparison": t.comparison,
                "baseline_values": t.baseline_values,
                "treatment_values": t.treatment_values,
                "mean_diff": t.mean_diff,
                "t_stat": t.t_stat,
                "p_value": t.p_value,
                "cohens_d": t.cohens_d,
                "ci_95": [t.ci_low, t.ci_high],
                "significant": bool(t.p_value < 0.05),
            }
            for t in tests
        ],
    }
