"""Write the power diagnostic that sits beside a gate decision.

Deliberately a separate artifact from ``gate_decision.json``. This number is
context for reading a verdict, never an input to one, and keeping it out of the
decision payload is what makes that claim testable rather than asserted.

Continuous regime only. A paired-binary companion was written and withdrawn: it
emitted values above the algebraic maximum ``|p01 - p10| <= p01 + p10`` across
this project's entire operating range, and the discordance it would have been fed
here — per-example judge differences that are almost never exactly equal — is not
the pass/fail disagreement such a model is about.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from evolution.core.stats import min_detectable_effect_paired

_FILENAME = "power_diagnostics.json"


def build_power_diagnostics(
    baseline_scores: list[float],
    evolved_scores: list[float],
    *,
    confidence: float = 0.90,
    power: float = 0.80,
    decision_rule: Optional[str] = None,
) -> dict:
    """What effect this sample size could, and could not, have detected.

    ``decision_rule`` is recorded because the reported alpha describes the
    *interval* rule. Some runs decide by other means — a point estimate against
    zero, or the closed-loop constraint, which discards the interval entirely —
    and reporting an alpha as though it governed those would describe a rule that
    never ran.
    """
    # Checked before the emptiness short-circuit below, or a mismatched pair with
    # an empty baseline would slip through while its mirror raises.
    if len(baseline_scores) != len(evolved_scores):
        raise ValueError(
            f"power diagnostics need paired arrays of equal length; got "
            f"{len(baseline_scores)} baseline vs {len(evolved_scores)} evolved"
        )
    n = len(baseline_scores)
    diffs = [e - b for b, e in zip(baseline_scores, evolved_scores)]
    out: dict = {
        "n_examples": n,
        "observed_mean_difference": (sum(diffs) / n) if n else 0.0,
        "decision_rule": decision_rule,
        "alpha_describes": "the lower bound of the paired bootstrap interval",
    }
    if n > 1:
        cont = min_detectable_effect_paired(diffs, confidence=confidence, power=power)
        cont["alpha_one_sided"] = round(cont["alpha_one_sided"], 6)
        out["continuous"] = cont
    return out


def write_power_diagnostics(
    output_dir: Optional[Path],
    baseline_scores: list[float],
    evolved_scores: list[float],
    *,
    confidence: float = 0.90,
    power: float = 0.80,
    decision_rule: Optional[str] = None,
) -> tuple[Optional[Path], Optional[dict]]:
    """Write the diagnostic beside the run's other artifacts, if there is a dir.

    Returns ``(path, payload)``; both are None when there is nothing to write. A
    missing file means "not computed" — runs that abort before scoring never
    reach here — and never "nothing to detect".
    """
    if output_dir is None or not baseline_scores:
        return None, None
    payload = build_power_diagnostics(
        baseline_scores, evolved_scores, confidence=confidence, power=power,
        decision_rule=decision_rule,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _FILENAME
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, payload


def format_power_line(payload: dict) -> str:
    """One console line: what the run could have seen, next to what it saw.

    Keeps the sign of the observed difference. For a gate that only ever
    certifies improvements, a regression reported as a bare magnitude "above" the
    detectable effect reads as a well-powered win — the sign is the one bit that
    must not be dropped.
    """
    cont = payload.get("continuous")
    if not cont:
        return "  power: too few examples to state a detectable effect"
    observed = payload.get("observed_mean_difference", 0.0)
    if cont["mde"] == 0.0 and observed == 0.0:
        # Identical arms: no variation to power a test on, and no effect to detect.
        # Strict "<" would render this as an effect *above* the detection floor.
        return (
            f"  power: n={cont['n']}, arms are identical — no variation between them, "
            "so there is nothing to detect and nothing detected"
        )
    if abs(observed) <= cont["mde"]:
        verdict = "below it — this sample could not have shown an effect that small"
    elif observed < 0:
        verdict = "above it, but negative — a detectable regression"
    else:
        verdict = "above it"
    return (
        f"  power: n={cont['n']}, smallest detectable effect "
        f"≥{cont['mde']:.3f} (one-sided α={cont['alpha_one_sided']:.3f}, "
        f"power={cont['power']:.2f}); observed Δ={observed:+.3f} is {verdict}"
    )
