"""The statistics core: the only place the printed numbers are decided."""

from __future__ import annotations

import math

import pytest

from model_bench.errors import UsageError
from model_bench.stats import (
    Summary,
    mean,
    median,
    percentile,
    stdev,
    summarize,
)


def test_mean_of_a_single_value_is_that_value() -> None:
    assert mean([4.25]) == 4.25


def test_mean_averages() -> None:
    assert mean([1.0, 2.0, 6.0]) == 3.0


def test_mean_uses_exact_summation() -> None:
    # Naive left-to-right addition of these loses the small terms; fsum
    # does not. Throughput is a ratio of sums, so this matters.
    values = [1e16, 1.0, -1e16, 1.0]
    assert mean(values) == 0.5


def test_mean_rejects_an_empty_sample() -> None:
    with pytest.raises(UsageError, match="empty sample"):
        mean([])


def test_mean_rejects_nan() -> None:
    with pytest.raises(UsageError, match="NaN"):
        mean([1.0, math.nan])


def test_median_of_odd_count_is_the_middle() -> None:
    assert median([5.0, 1.0, 3.0]) == 3.0


def test_median_of_even_count_averages_the_middle_pair() -> None:
    assert median([4.0, 1.0, 3.0, 2.0]) == 2.5


def test_median_does_not_require_sorted_input() -> None:
    assert median([9.0, 0.0, 5.0]) == 5.0


def test_median_rejects_an_empty_sample() -> None:
    with pytest.raises(UsageError):
        median([])


def test_percentile_zero_is_the_minimum() -> None:
    assert percentile([3.0, 1.0, 2.0], 0.0) == 1.0


def test_percentile_one_is_the_maximum() -> None:
    assert percentile([3.0, 1.0, 2.0], 1.0) == 3.0


def test_percentile_interpolates_between_ranks() -> None:
    # p90 of 1..5 sits 3.6 of the way along, i.e. between 4 and 5.
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.90) == pytest.approx(4.6)


def test_percentile_lands_exactly_on_a_rank_when_it_can() -> None:
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0


def test_percentile_of_one_value_is_that_value() -> None:
    assert percentile([7.5], 0.9) == 7.5


def test_percentile_rejects_a_fraction_above_one() -> None:
    with pytest.raises(UsageError, match=r"\[0, 1\]"):
        percentile([1.0], 90)


def test_percentile_rejects_a_negative_fraction() -> None:
    with pytest.raises(UsageError):
        percentile([1.0], -0.1)


def test_percentile_rejects_an_empty_sample() -> None:
    with pytest.raises(UsageError):
        percentile([], 0.5)


def test_stdev_is_none_for_a_single_measurement() -> None:
    # One run tells you nothing about spread; 0.0 would claim it did.
    assert stdev([1.0]) is None


def test_stdev_is_the_sample_deviation_not_the_population_one() -> None:
    # Population stdev of [2, 4] is 1.0; sample stdev is sqrt(2).
    result = stdev([2.0, 4.0])
    assert result == pytest.approx(math.sqrt(2.0))


def test_stdev_of_identical_values_is_zero() -> None:
    assert stdev([3.0, 3.0, 3.0]) == 0.0


def test_stdev_rejects_an_empty_sample() -> None:
    with pytest.raises(UsageError):
        stdev([])


def test_summarize_reports_every_field() -> None:
    summary = summarize([10.0, 20.0, 30.0, 40.0])
    assert summary.count == 4
    assert summary.mean == 25.0
    assert summary.median == 25.0
    assert summary.minimum == 10.0
    assert summary.maximum == 40.0
    assert summary.p90 == pytest.approx(37.0)
    assert summary.stdev == pytest.approx(12.909944, rel=1e-6)


def test_summarize_of_one_value_has_no_stdev() -> None:
    summary = summarize([2.0])
    assert summary.count == 1
    assert summary.stdev is None
    assert summary.minimum == summary.maximum == 2.0


def test_summarize_rejects_an_empty_sample() -> None:
    with pytest.raises(UsageError):
        summarize([])


def test_spread_is_max_minus_min() -> None:
    assert summarize([1.0, 5.0, 3.0]).spread == 4.0


def test_relative_stdev_is_a_fraction_of_the_mean() -> None:
    summary = summarize([9.0, 11.0])
    assert summary.relative_stdev == pytest.approx(summary.stdev / 10.0)  # type: ignore[operator]


def test_relative_stdev_is_none_without_a_stdev() -> None:
    assert summarize([4.0]).relative_stdev is None


def test_relative_stdev_is_none_when_the_mean_is_zero() -> None:
    # Not infinity: an infinity would be formatted into a report.
    assert summarize([-1.0, 1.0]).relative_stdev is None


def test_relative_stdev_uses_the_absolute_mean() -> None:
    summary = summarize([-9.0, -11.0])
    assert summary.relative_stdev is not None
    assert summary.relative_stdev > 0


def test_as_dict_is_json_ready() -> None:
    payload = summarize([1.0, 2.0]).as_dict()
    assert set(payload) == {"count", "mean", "median", "min", "max", "p90", "stdev"}
    assert payload["count"] == 2


def test_as_dict_keeps_a_missing_stdev_as_null() -> None:
    assert summarize([1.0]).as_dict()["stdev"] is None


def test_summary_is_frozen() -> None:
    summary = summarize([1.0, 2.0])
    with pytest.raises(AttributeError):
        summary.mean = 99.0  # type: ignore[misc]


def test_summary_can_be_constructed_directly() -> None:
    summary = Summary(
        count=1, mean=1.0, median=1.0, minimum=1.0, maximum=1.0, p90=1.0, stdev=None
    )
    assert summary.spread == 0.0


def test_integers_are_accepted_and_widened() -> None:
    assert summarize([1, 2, 3]).mean == 2.0


def test_percentile_handles_a_two_value_sample() -> None:
    assert percentile([0.0, 10.0], 0.25) == pytest.approx(2.5)
