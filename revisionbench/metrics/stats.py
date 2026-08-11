"""Bootstrap intervals and the percentile convention they rest on (plan.md §11).

plan.md §11 asks for bootstrap CIs rather than bare means on every reported aggregate.
Two things here are worth stating once so that every call site inherits them:

**The averaging convention is fixed and tested.** A "median" implemented as
``sorted(x)[len(x) // 2]`` is the *upper* of the two middle values on an even-length
sample. MirrorBench shipped exactly that and it printed 0.09 where the median was 0.05,
on a bimodal distribution where the difference mattered. :func:`percentile` interpolates
linearly between order statistics, :func:`median` is defined in terms of it, and nothing
in this repo should compute either by hand.

**Bootstrap resampling units are the caller's problem.** :func:`bootstrap_ci` resamples
whatever sequence it is given. For revision-bench the natural unit is the *passage*, not
the round and not the sentence: rounds within a passage are a dependent series by
construction (round k+1 is a revision of round k), so resampling rounds would treat one
trajectory as many independent observations and return an interval that is far too
narrow. plan.md §11's mixed-effects structure — rounds nested in passages nested in
authors — is the same statement. When in doubt, resample passages.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence

__all__ = [
    "bootstrap_ci",
    "mean",
    "median",
    "percentile",
    "stdev",
]


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean.

    Raises:
        ValueError: ``values`` is empty. An empty cell has no mean, and returning 0.0
            would put a measured-looking number where there is no measurement.
    """
    if not values:
        raise ValueError("mean of an empty sequence is undefined; report the cell as missing")
    return sum(float(v) for v in values) / len(values)


def stdev(values: Sequence[float], *, sample: bool = True) -> float:
    """Standard deviation.

    Args:
        values: The observations.
        sample: Use the ``n-1`` denominator (Bessel's correction). ``True`` is right when
            the values are a sample from a wider population — the usual case here, where
            10 passages stand in for "literary prose". Pass ``False`` only when the values
            *are* the whole population.

    Raises:
        ValueError: Fewer than two values with ``sample=True``, or none at all. A
            one-element sample has no spread to estimate; 0.0 would claim it does.
    """
    n = len(values)
    if n == 0:
        raise ValueError("stdev of an empty sequence is undefined")
    if sample and n < 2:
        raise ValueError(
            "sample stdev needs at least two values; with one observation the spread is "
            "unmeasured, not zero"
        )
    mu = mean(values)
    denominator = n - 1 if sample else n
    return (sum((float(v) - mu) ** 2 for v in values) / denominator) ** 0.5


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence.

    Args:
        sorted_values: Values in ascending order. Not sorted here: callers that already
            sorted (the bootstrap does) should not pay for it twice, and a caller that
            forgets is a bug worth surfacing in a test rather than papering over.
        q: Quantile in ``[0, 1]``.

    Raises:
        ValueError: The sequence is empty, or ``q`` is outside ``[0, 1]``.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return float(sorted_values[lower]) * (1.0 - weight) + float(sorted_values[upper]) * weight


def median(values: Sequence[float]) -> float:
    """Median, averaging the two middle values on an even-length sample.

    Defined via :func:`percentile` so that the convention is stated in exactly one place.
    """
    return percentile(sorted(float(v) for v in values), 0.5)


def bootstrap_ci[T](
    values: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for ``statistic`` over ``values``.

    Draws ``n_resamples`` resamples with replacement (each the same length as ``values``),
    evaluates ``statistic`` on each, and returns the ``alpha/2`` and ``1 - alpha/2``
    percentiles of the resulting distribution.

    ``statistic`` must be defined on *every* resample. Silently dropping degenerate
    resamples would bias the interval, so this function does not do it; wrap the statistic
    if it can fail.

    Reproducibility: identical ``values``, ``statistic``, ``n_resamples`` and ``seed``
    give bit-identical bounds. A local :class:`random.Random` does the drawing, so global
    RNG state is neither read nor disturbed.

    Args:
        values: Resampling units — see the module docstring on choosing them. A single
            value gives a zero-width interval, which is the bootstrap reporting honestly
            on one observation rather than a failure.
        statistic: Callable over a resampled sequence, returning a float.
        n_resamples: How many resamples to draw. Keep at or above 1000 for anything
            reported (plan.md §11).
        seed: Seed for the local RNG.
        alpha: Two-sided miss rate; the default 0.05 gives a 95% interval.

    Returns:
        ``(lower, upper)``.

    Raises:
        ValueError: ``values`` is empty, ``n_resamples`` is below 1, or ``alpha`` is not
            strictly between 0 and 1.
    """
    if not values:
        raise ValueError(
            "bootstrap_ci needs at least one value to resample; an empty cell has no "
            "interval, so report it as missing"
        )
    if n_resamples < 1:
        raise ValueError(
            f"bootstrap_ci needs n_resamples >= 1, got {n_resamples}; plan.md §11 asks "
            f"for at least 1000 on anything reported"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")

    rng = random.Random(seed)
    n = len(values)
    estimates = sorted(float(statistic(rng.choices(values, k=n))) for _ in range(n_resamples))
    return (percentile(estimates, alpha / 2.0), percentile(estimates, 1.0 - alpha / 2.0))
