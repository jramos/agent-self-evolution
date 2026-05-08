#!/usr/bin/env bash
# Resume Stage 5 of the deploy-gate calibration campaign after the
# apple-notes crash on 2026-05-07. The original runbook (run_campaign.sh)
# would re-run the smoke + nano-pdf seed=7 if invoked again from
# SKIP_TO_STAGE=5 — wasteful given those already completed successfully.
#
# This script picks up exactly at apple-notes seed=42 (the run that
# crashed) and runs only the 10 remaining invocations: 6 growth + 4
# control. Same env-var contract as the parent runbook.
#
# Usage:
#     ulimit -n 65536          # belt-and-suspenders against the asyncio FD leak
#     CAMPAIGN_START_TS=20260507_134805 \
#         N_STAR=250 RATIO_STAR=0.65 \
#         bash scripts/resume_stage5.sh
#
# Resume mid-array (e.g. after a JSON-parse crash on huggingface-hub):
#     SKIP_GROWTH=4 bash scripts/resume_stage5.sh
#         # ...skips the first 4 entries (apple-notes 42, apple-notes 7,
#         # polymarket 42, polymarket 7) and starts at huggingface-hub 42.
#     SKIP_CONTROLS=2 bash scripts/resume_stage5.sh
#         # ...starts the control block at index 2.

set -euo pipefail

: "${CAMPAIGN_START_TS:?CAMPAIGN_START_TS required (e.g. 20260507_134805)}"
: "${N_STAR:?N_STAR required (e.g. 250)}"
: "${RATIO_STAR:?RATIO_STAR required (e.g. 0.65)}"

CAP_USD=200
STATUS_CMD="uv run python scripts/campaign_status.py --since ${CAMPAIGN_START_TS} --cap ${CAP_USD}"
EVOLVE="uv run python -m evolution.skills.evolve_skill --iterations 10 --evaluate-band-on-holdout"

confirm() {
    local prompt="$1"
    read -r -p "${prompt} [y/N] " answer
    case "${answer}" in
        y|Y|yes|YES) return 0 ;;
        *) echo "Aborted."; exit 1 ;;
    esac
}

# Remaining growth runs (apple-notes seed=42 onward — nano-pdf seed=42
# and seed=7 were already produced; the smoke at 134805 satisfied the
# growth-pct>0 gate).
declare -a REMAINING_GROWTH=(
    "apple-notes 42"
    "apple-notes 7"
    "polymarket 42"
    "polymarket 7"
    "huggingface-hub 42"
    "huggingface-hub 7"
)

# Control runs (1 seed each at --quality-gate default — confirms
# BAP-off is the variable causing growth, not anything else).
# huggingface-hub dropped from the corpus on 2026-05-08: the synthetic
# generator hit a 92% case-filter drop rate at N=250, leaving 9 holdout
# examples (below min_holdout_size=10). Two cache-stuck malformed-JSON
# crashes preceded the holdout-size failure. Future campaigns should
# either swap this skill or use a different eval source.
declare -a CONTROL_PAIRS=(
    "nano-pdf 42"
    "apple-notes 42"
    "polymarket 42"
)

SKIP_GROWTH="${SKIP_GROWTH:-0}"
SKIP_CONTROLS="${SKIP_CONTROLS:-0}"

# Slice the arrays by SKIP offset (bash 4+ array slicing).
REMAINING_GROWTH=("${REMAINING_GROWTH[@]:${SKIP_GROWTH}}")
CONTROL_PAIRS=("${CONTROL_PAIRS[@]:${SKIP_CONTROLS}}")

echo "=== Resume Stage 5 ==="
echo "Campaign start:    ${CAMPAIGN_START_TS}"
echo "N* / ratio*:       ${N_STAR} / ${RATIO_STAR}"
echo "ulimit -n (FDs):   $(ulimit -n)"
echo "SKIP_GROWTH:       ${SKIP_GROWTH}"
echo "SKIP_CONTROLS:     ${SKIP_CONTROLS}"
echo "Remaining growth:  ${#REMAINING_GROWTH[@]} → ${REMAINING_GROWTH[*]:-(none)}"
echo "Remaining control: ${#CONTROL_PAIRS[@]} → ${CONTROL_PAIRS[*]:-(none)}"
echo
${STATUS_CMD}
echo

if [[ "${#REMAINING_GROWTH[@]}" -gt 0 ]]; then
    first_growth="${REMAINING_GROWTH[0]}"
    confirm "Proceed with ${first_growth} (growth, lenient + bap-safety=0.0)?"
fi

for entry in "${REMAINING_GROWTH[@]}"; do
    read -r skill seed <<<"${entry}"
    echo
    echo ">>> GROWTH: ${skill} seed=${seed}"
    ${EVOLVE} \
        --skill "${skill}" --seed "${seed}" \
        --quality-gate lenient --bap-safety-margin 0.0 \
        --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
    ${STATUS_CMD}
    confirm "Continue Stage 5?"
done

echo
echo "--- Control runs (--quality-gate default, no BAP override) ---"
confirm "Proceed with control runs?"
for entry in "${CONTROL_PAIRS[@]}"; do
    read -r skill seed <<<"${entry}"
    echo
    echo ">>> CONTROL: ${skill} seed=${seed} (--quality-gate default)"
    ${EVOLVE} \
        --skill "${skill}" --seed "${seed}" \
        --quality-gate default \
        --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
done
${STATUS_CMD}

echo
echo "=== Stage 5 complete ==="
echo "Run analysis:"
echo "  uv run python scripts/analysis/study_b_pick_epsilon.py"
echo "  uv run python scripts/analysis/study_c_pick_curve.py"
echo "Then resume the runbook at Stage 6 onward:"
echo "  CAMPAIGN_START_TS=${CAMPAIGN_START_TS} \\"
echo "      N_STAR=${N_STAR} RATIO_STAR=${RATIO_STAR} \\"
echo "      EPSILON_MULTIPLIER=<from study B> \\"
echo "      FREE_STAR=<from study C> SLOPE_STAR=<from study C> \\"
echo "      SKIP_TO_STAGE=7 \\"
echo "      bash scripts/run_campaign.sh"
