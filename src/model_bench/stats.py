"""Summary statistics over benchmark samples.

This module is deliberately pure: it takes floats and returns floats,
knows nothing about HTTP, models or time, and is where the numbers a
user actually reads are decided. Every claim the reports make is
computed here, so this is the part that is unit-tested hardest.

Two choices worth spelling out.

**Percentiles use linear interpolation between the two nearest ranks**
(the same rule as ``numpy.percentile``'s default and Excel's
``PERCENTILE.INC``). Nearest-rank would be defensible too, but with the
three-to-five repetitions a benchmark run typically has, nearest-rank
makes p90 and p100 the same number, which reads as suspiciously tidy.

**Standard deviation is the sample deviation (n-1)** because the
repetitions are a sample of the runs the machine could have done, not
the population of every run it ever will. With a single repetition it
is ``None``, not ``0.0``: one measurement carries no information about
spread, and printing ``±0.00`` after one run would claim otherwise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .errors import UsageError


@dataclass(frozen=True)
class Summary:
    """The distribution of one metric over the repetitions of one prompt."""

    count: int
    mean: float
    median: float
    minimum: float
    maximum: float
    p90: float
    stdev: float | None

    @property
    def spread(self) -> float:
        """Max minus min — the crudest, most honest measure of noise."""
        return self.maximum - self.minimum

    @property
    def relative_stdev(self) -> float | None:
        """Standard deviation as a fraction of the mean.

        ``None`` when there is nothing to divide by (a single sample, or
        a mean of zero), rather than an infinity that would then be
        formatted into a report.
        """
        if self.stdev is None or self.mean == 0.0:
            return None
        return self.stdev / abs(self.mean)

    def as_dict(self) -> dict[str, float | int | None]:
        """A JSON-ready mapping, used by the ``json`` report format."""
        return {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "p90": self.p90,
            "stdev": self.stdev,
        }


def _validated(values: Iterable[float]) -> list[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise UsageError("cannot summarise an empty sample")
    for value in ordered:
        if math.isnan(value):
            raise UsageError("cannot summarise a sample containing NaN")
    return ordered


def mean(values: Iterable[float]) -> float:
    """The arithmetic mean of a non-empty sample."""
    ordered = _validated(values)
    return math.fsum(ordered) / len(ordered)


def median(values: Iterable[float]) -> float:
    """The middle value, averaging the middle pair for even counts."""
    ordered = _validated(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def percentile(values: Iterable[float], fraction: float) -> float:
    """The ``fraction`` percentile, interpolating between ranks.

    ``fraction`` is in ``[0, 1]``; ``0`` is the minimum and ``1`` the
    maximum. A fraction outside that range is a programming error, not a
    clamped request, so it raises.
    """
    if not 0.0 <= fraction <= 1.0:
        raise UsageError(f"percentile fraction must be within [0, 1], got {fraction}")
    ordered = _validated(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stdev(values: Iterable[float]) -> float | None:
    """Sample standard deviation, or ``None`` for a single measurement."""
    ordered = _validated(values)
    if len(ordered) < 2:
        return None
    centre = math.fsum(ordered) / len(ordered)
    variance = math.fsum((value - centre) ** 2 for value in ordered) / (len(ordered) - 1)
    return math.sqrt(variance)


def summarize(values: Sequence[float]) -> Summary:
    """Every statistic the reports use, computed once over one sample."""
    ordered = _validated(values)
    return Summary(
        count=len(ordered),
        mean=math.fsum(ordered) / len(ordered),
        median=median(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        p90=percentile(ordered, 0.90),
        stdev=stdev(ordered),
    )
