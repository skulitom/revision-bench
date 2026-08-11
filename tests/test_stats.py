"""Statistics conventions: the averaging rule and bootstrap reproducibility."""

from __future__ import annotations

import pytest

from revisionbench.metrics.stats import bootstrap_ci, mean, median, percentile, stdev


class TestMedian:
    def test_even_length_averages_the_two_middle_values(self) -> None:
        """Not `sorted(x)[len(x) // 2]`, which is the *upper* middle value.

        MirrorBench shipped that version and it printed 0.09 where the median was 0.05 on
        a bimodal sample. The convention is defined once, here, and tested.
        """
        assert median([1.0, 2.0, 3.0, 4.0]) == 2.5
        assert median([0.0, 0.0, 1.0, 1.0]) == 0.5

    def test_odd_length(self) -> None:
        assert median([3.0, 1.0, 2.0]) == 2.0

    def test_single_value(self) -> None:
        assert median([7.0]) == 7.0


class TestPercentile:
    def test_endpoints(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0]
        assert percentile(values, 0.0) == 1.0
        assert percentile(values, 1.0) == 4.0

    def test_interpolates(self) -> None:
        assert percentile([0.0, 10.0], 0.25) == 2.5

    def test_out_of_range_q(self) -> None:
        with pytest.raises(ValueError, match=r"q must be in \[0, 1\]"):
            percentile([1.0], 1.5)

    def test_empty(self) -> None:
        with pytest.raises(ValueError, match="empty sequence"):
            percentile([], 0.5)


class TestMeanAndStdev:
    def test_mean(self) -> None:
        assert mean([1.0, 2.0, 3.0]) == 2.0

    def test_empty_mean_raises_rather_than_returning_zero(self) -> None:
        with pytest.raises(ValueError, match="report the cell as missing"):
            mean([])

    def test_sample_stdev(self) -> None:
        assert stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) == pytest.approx(2.13809, rel=1e-4)

    def test_population_stdev(self) -> None:
        assert stdev([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], sample=False) == pytest.approx(2.0)

    def test_single_value_sample_stdev_raises(self) -> None:
        """0.0 would claim a spread was measured when it was not."""
        with pytest.raises(ValueError, match="at least two values"):
            stdev([1.0])


class TestBootstrap:
    def test_is_reproducible_for_a_given_seed(self) -> None:
        values = [float(v) for v in range(20)]
        a = bootstrap_ci(values, mean, n_resamples=200, seed=7)
        b = bootstrap_ci(values, mean, n_resamples=200, seed=7)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        values = [float(v) for v in range(20)]
        assert bootstrap_ci(values, mean, n_resamples=200, seed=1) != bootstrap_ci(
            values, mean, n_resamples=200, seed=2
        )

    def test_interval_brackets_the_point_estimate(self) -> None:
        values = [float(v) for v in range(50)]
        low, high = bootstrap_ci(values, mean, n_resamples=1000, seed=0)
        assert low < mean(values) < high

    def test_single_observation_gives_a_zero_width_interval(self) -> None:
        """Honest reporting on one observation, not a failure."""
        low, high = bootstrap_ci([4.0], mean, n_resamples=50, seed=0)
        assert low == high == 4.0

    def test_does_not_disturb_global_rng_state(self) -> None:
        import random

        random.seed(1234)
        expected = random.random()
        random.seed(1234)
        bootstrap_ci([1.0, 2.0, 3.0], mean, n_resamples=50, seed=99)
        assert random.random() == expected

    def test_empty_values_raise(self) -> None:
        with pytest.raises(ValueError, match="report it as missing"):
            bootstrap_ci([], mean)

    def test_too_few_resamples_raise(self) -> None:
        with pytest.raises(ValueError, match="n_resamples >= 1"):
            bootstrap_ci([1.0], mean, n_resamples=0)

    def test_bad_alpha_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly between 0 and 1"):
            bootstrap_ci([1.0], mean, alpha=1.0)

    def test_a_failing_statistic_propagates(self) -> None:
        """Silently dropping degenerate resamples would bias the interval."""

        def picky(sample):
            if len(set(sample)) < 2:
                raise ValueError("degenerate resample")
            return mean(sample)

        with pytest.raises(ValueError, match="degenerate resample"):
            bootstrap_ci([1.0, 1.0], picky, n_resamples=10, seed=0)
