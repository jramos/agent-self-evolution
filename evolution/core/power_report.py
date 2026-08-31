"""Write the power diagnostics that sit beside a gate decision.

Deliberately a separate artifact from ``gate_decision.json``. These numbers are
context for reading a verdict, never an input to one, and keeping them out of the
decision payload is what makes that claim testable rather than asserted.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

from evolution.core.stats import (
    min_detectable_effect_paired,
    min_detectable_shift_paired_binary,
)

_FILENAME = "power_diagnostics.json"


def build_power_diagnostics(
    baseline_scores: list[float],
    evolved_scores: list[float],
    *,
    confidence: float = 0.90,
    power: float = 0.80,
) -> dict:
    """What effect this sample size could and could not have detected.

    Both regimes are reported from the same paired arrays: the continuous one
    from the spread of the per-example differences, and the paired-binary one
    from how often the two arms actually disagree — which is what paired-binary
    power depends on, unlike the marginal pass rate.
    """
    n = len(baseline_scores)
    diffs = [e - b for b, e in zip(baseline_scores, evolved_scores)]
    sd_diff = statistics.stdev(diffs) if n > 1 else 0.0
    discordant = sum(1 for d in diffs if d != 0)

    out: dict = {
        "n_examples": n,
        "observed_mean_difference": (sum(diffs) / n) if n else 0.0,
        "discordant_pairs": discordant,
    }
    if n > 1:
        out["continuous"] = min_detectable_effect_paired(
            n, sd_diff, confidence=confidence, power=power
        )
        if discordant:
            out["paired_binary"] = min_detectable_shift_paired_binary(
                n, discordance_rate=discordant / n, confidence=confidence, power=power
            )
        else:
            # Every pair agreed: there is no discordance to power a paired-binary
            # test on, and inventing a rate to report one would be worse than
            # saying so.
            out["paired_binary"] = None
    return out


def write_power_diagnostics(
    output_dir: Optional[Path],
    baseline_scores: list[float],
    evolved_scores: list[float],
    *,
    confidence: float = 0.90,
    power: float = 0.80,
) -> Optional[Path]:
    """Write the diagnostics beside the run's other artifacts, if there is a dir.

    Returns the path written, or None. Absent on runs that abort before scoring —
    consumers should treat a missing file as "not computed", not as "nothing to
    detect".
    """
    if output_dir is None or not baseline_scores:
        return None
    payload = build_power_diagnostics(
        baseline_scores, evolved_scores, confidence=confidence, power=power
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _FILENAME
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_power_line(payload: dict) -> str:
    """One console line: what the run could have seen, next to what it saw."""
    cont = payload.get("continuous")
    if not cont:
        return "  power: too few examples to state a detectable effect"
    observed = abs(payload.get("observed_mean_difference", 0.0))
    verdict = "below" if observed < cont["mde"] else "above"
    return (
        f"  power: n={payload['n_examples']}, smallest detectable effect "
        f"≥{cont['mde']:.3f} (one-sided α={cont['alpha_one_sided']:.3f}, "
        f"power={cont['power']:.2f}); observed |Δ|={observed:.3f} is {verdict} it"
    )
