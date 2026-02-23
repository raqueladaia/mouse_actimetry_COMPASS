"""Statistical tests for group comparisons."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import scipy.stats as stats
from statsmodels.stats.multitest import fdrcorrection


def welch_ttest(
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[float, float]:
    """Welch's t-test (unequal variances) between two groups.

    Parameters
    ----------
    group1, group2 : np.ndarray
        1-D arrays of values for each group.

    Returns
    -------
    t_stat : float
    p_value : float
    """
    result = stats.ttest_ind(group1, group2, equal_var=False)
    return float(result.statistic), float(result.pvalue)


def mann_whitney(
    group1: np.ndarray,
    group2: np.ndarray,
) -> tuple[float, float]:
    """Mann-Whitney U test between two groups.

    Parameters
    ----------
    group1, group2 : np.ndarray

    Returns
    -------
    u_stat : float
    p_value : float
    """
    result = stats.mannwhitneyu(group1, group2)
    return float(result.statistic), float(result.pvalue)


def fdr_correct(
    p_values: np.ndarray,
    method: str = "p",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply FDR (Benjamini-Hochberg) correction to a set of p-values.

    Parameters
    ----------
    p_values : np.ndarray
        Array of raw p-values.
    method : str
        Method for ``fdrcorrection`` (default ``"p"`` for p-value based).

    Returns
    -------
    rejected : np.ndarray of bool
        Whether each null hypothesis is rejected.
    q_values : np.ndarray
        Corrected p-values.
    """
    return fdrcorrection(p_values, method=method)


def pairwise_fdr_tests(
    group_data: dict[str, np.ndarray | list],
    test: str = "welch",
) -> list[tuple[str, str, float, bool]]:
    """Run all pairwise comparisons with Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    group_data : dict[str, np.ndarray | list]
        Mapping of group name to per-animal values (1-D array or list).
    test : str
        ``"welch"`` for Welch's t-test or ``"mannwhitney"`` for Mann-Whitney U.

    Returns
    -------
    list of (group1, group2, p_fdr, significant)
        One entry per pair.  ``p_fdr`` is the FDR-corrected p-value and
        ``significant`` is ``True`` when ``p_fdr < 0.05``.
    """
    groups = list(group_data.keys())
    pairs = list(combinations(groups, 2))
    if not pairs:
        return []

    raw_p: list[float] = []
    for g1, g2 in pairs:
        a1 = np.asarray(group_data[g1], dtype=float)
        a2 = np.asarray(group_data[g2], dtype=float)
        # Skip degenerate cases (empty or constant arrays)
        if len(a1) < 2 or len(a2) < 2:
            raw_p.append(1.0)
            continue
        if test == "welch":
            _, p = welch_ttest(a1, a2)
        else:
            _, p = mann_whitney(a1, a2)
        raw_p.append(p)

    _, q = fdr_correct(np.array(raw_p))
    return [
        (g1, g2, float(q[i]), bool(q[i] < 0.05))
        for i, (g1, g2) in enumerate(pairs)
    ]


def bootstrap_mean_difference(
    data1: list[np.ndarray],
    data2: list[np.ndarray],
    n_bootstrap: int = 10_000,
    ci: float = 95,
    sample_fraction: float = 0.7,
) -> dict:
    """Bootstrap test for the difference in means between two groups.

    Each group is a list of arrays (e.g., one array per day or cycle).
    All arrays are flattened before sampling.

    Parameters
    ----------
    data1, data2 : list of np.ndarray
        Each element is an array of bout durations (one per day/cycle).
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence interval percentage (e.g., 95).
    sample_fraction : float
        Fraction of the smaller group to sample per iteration.

    Returns
    -------
    dict
        Keys: ``observed_diff``, ``bootstrap_diffs``, ``lower_ci``,
        ``upper_ci``, ``p_value``.

        ``p_value`` is a two-tailed bootstrap p-value: the proportion of
        bootstrap diffs with opposite sign to the observed diff, multiplied
        by 2, floored at ``1 / n_bootstrap``.
    """
    flat1 = np.hstack([np.hstack(d) for d in data1])
    flat2 = np.hstack([np.hstack(d) for d in data2])

    sample_size = int(np.ceil(min(len(flat1), len(flat2)) * sample_fraction))
    observed_diff = float(np.mean(flat1) - np.mean(flat2))

    rng = np.random.default_rng()
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        s1 = rng.choice(flat1, size=sample_size, replace=True)
        s2 = rng.choice(flat2, size=sample_size, replace=True)
        boot_diffs[i] = np.mean(s1) - np.mean(s2)

    lower = float(np.percentile(boot_diffs, (100 - ci) / 2))
    upper = float(np.percentile(boot_diffs, 100 - (100 - ci) / 2))

    # Two-tailed bootstrap p-value
    if observed_diff > 0:
        n_opposite = int(np.sum(boot_diffs < 0))
    elif observed_diff < 0:
        n_opposite = int(np.sum(boot_diffs > 0))
    else:
        n_opposite = n_bootstrap
    p_value = min(1.0, 2 * n_opposite / n_bootstrap)
    p_value = max(p_value, 1.0 / n_bootstrap)

    return {
        "observed_diff": observed_diff,
        "bootstrap_diffs": boot_diffs,
        "lower_ci": lower,
        "upper_ci": upper,
        "p_value": p_value,
    }


def pairwise_bootstrap_fdr_tests(
    group_data: dict[str, np.ndarray | list],
    n_bootstrap: int = 10_000,
    ci: float = 95,
    sample_fraction: float = 0.7,
) -> list[tuple[str, str, float, bool, dict]]:
    """Run pairwise bootstrap comparisons with FDR correction.

    Parameters
    ----------
    group_data : dict[str, np.ndarray | list]
        Mapping of group name to a flat array/list of pooled values
        (e.g., bout durations pooled across animals for a given phase).
    n_bootstrap : int
        Number of bootstrap iterations per pair.
    ci : float
        Confidence interval percentage.
    sample_fraction : float
        Fraction of the smaller group to sample per iteration.

    Returns
    -------
    list of (group1, group2, p_fdr, significant, result_dict)
        One entry per pair.  ``p_fdr`` is the FDR-corrected p-value,
        ``significant`` is ``True`` when ``p_fdr < 0.05``, and
        ``result_dict`` is the full bootstrap result dict.
    """
    groups = list(group_data.keys())
    pairs = list(combinations(groups, 2))
    if not pairs:
        return []

    raw_p: list[float] = []
    results: list[dict] = []
    for g1, g2 in pairs:
        a1 = np.asarray(group_data[g1], dtype=float)
        a2 = np.asarray(group_data[g2], dtype=float)
        if len(a1) < 2 or len(a2) < 2:
            raw_p.append(1.0)
            results.append({
                "observed_diff": 0.0, "bootstrap_diffs": np.array([]),
                "lower_ci": 0.0, "upper_ci": 0.0, "p_value": 1.0,
            })
            continue
        # Wrap flat arrays as single-element lists for bootstrap_mean_difference
        res = bootstrap_mean_difference(
            [a1], [a2],
            n_bootstrap=n_bootstrap, ci=ci, sample_fraction=sample_fraction,
        )
        raw_p.append(res["p_value"])
        results.append(res)

    _, q = fdr_correct(np.array(raw_p))
    return [
        (g1, g2, float(q[i]), bool(q[i] < 0.05), results[i])
        for i, (g1, g2) in enumerate(pairs)
    ]
