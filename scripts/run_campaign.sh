#!/usr/bin/env bash
# Deploy-gate calibration campaign runbook.
#
# This is a checkpoint-driven driver, not a fire-and-forget script: 24
# evolve runs at ~$10 each is too much money to launch unattended. Each
# phase pauses for explicit confirmation; cost monitoring happens via
# `scripts/campaign_status.py` between phases.
#
# Stages map to ~/.claude/plans/i-want-to-implement-peaceful-stearns.md.
#
# Pre-reqs:
#   - Stage 1 code prep merged (this branch).
#   - SKILL_SOURCES_HERMES_REPO set so --skill resolves to the hermes-agent
#     skills directory.
#   - OPENAI_API_KEY populated.
#
# Usage:
#   bash scripts/run_campaign.sh
#
# Resume hints (set on the same line as the bash invocation):
#   CAMPAIGN_START_TS=20260507_120000 \
#       SKIP_DATASET_GEN=1 \
#       N_STAR=250 RATIO_STAR=0.65 \
#       SKIP_TO_STAGE=5 \
#       bash scripts/run_campaign.sh
#
# Skip flags:
#   SKIP_DATASET_GEN=1   — skip Stage 4's $20 dataset gen loop (already ran)
#   SKIP_TO_STAGE=5|6|7  — jump in at the named stage (skip earlier stages)
#   N_STAR / RATIO_STAR  — pre-set so the script doesn't prompt
#   EPSILON_MULTIPLIER   — pre-set ε multiplier (Stage 6)
#   FREE_STAR / SLOPE_STAR — pre-set Study C output (Stage 6); use "KEEP_CURRENT" for FREE_STAR if Study C returned INSUFFICIENT_DATA

set -euo pipefail

CAMPAIGN_START_TS="${CAMPAIGN_START_TS:-$(date +%Y%m%d_%H%M%S)}"
CAP_USD=200
SKIP_TO_STAGE="${SKIP_TO_STAGE:-4}"
STATUS_CMD="uv run python scripts/campaign_status.py --since ${CAMPAIGN_START_TS} --cap ${CAP_USD}"
EVOLVE="uv run python -m evolution.skills.evolve_skill --iterations 10 --evaluate-band-on-holdout"

# Calibration corpus (Stage 2)
SKILLS_CALIB_SMALL=(nano-pdf apple-notes)
SKILLS_CALIB_MED=(polymarket huggingface-hub)
SKILLS_CALIB_LARGE=(notion spotify)
SKILLS_CALIB_ALL=("${SKILLS_CALIB_SMALL[@]}" "${SKILLS_CALIB_MED[@]}" "${SKILLS_CALIB_LARGE[@]}")

# Validation corpus (Stage 7) — must NOT appear in output/ at campaign start.
# `plan` (1981 chars) dropped on 2026-05-08: synthetic generator produced
# only 16 valid cases → 7 holdout < min_holdout_size=10. Same failure mode
# observed on the small-bucket calibration skills (nano-pdf 78% drop,
# apple-notes 88%). Future campaigns should use a different judge or
# inflate eval_dataset_size for skills under 2500 chars.
SKILLS_VALIDATION=(maps linear)

confirm() {
    local prompt="$1"
    read -r -p "${prompt} [y/N] " answer
    case "${answer}" in
        y|Y|yes|YES) return 0 ;;
        *) echo "Aborted."; exit 1 ;;
    esac
}

prompt_if_unset() {
    # prompt_if_unset VAR_NAME "prompt text"
    local var_name="$1"
    local prompt_text="$2"
    if [[ -z "${!var_name:-}" ]]; then
        read -r -p "${prompt_text}: " value
        if [[ -z "${value}" ]]; then
            echo "Aborted: ${var_name} required."; exit 1
        fi
        printf -v "${var_name}" '%s' "${value}"
    fi
}

stage_active() {
    # stage_active STAGE_NUM — true when current stage >= SKIP_TO_STAGE
    [[ "${SKIP_TO_STAGE}" -le "$1" ]]
}

echo "=== Deploy-gate calibration campaign ==="
echo "Campaign start: ${CAMPAIGN_START_TS}"
echo "Cost cap:       \$${CAP_USD}"
echo "Skip-to-stage:  ${SKIP_TO_STAGE}"
echo

# ----------------------------------------------------------------------
# Stage 4 — Study A: generate large holdout pools, then pick N* and ratio*
# ----------------------------------------------------------------------
if stage_active 4; then
    echo "--- Stage 4: Study A (dataset gen + N*/ratio* analysis) ---"
    if [[ "${SKIP_DATASET_GEN:-0}" != "1" ]]; then
        confirm "Generate N=400 holdout pools for the 6 calibration skills (~\$20)?"
        for skill in "${SKILLS_CALIB_ALL[@]}"; do
            echo ">>> Generating pool for ${skill}"
            uv run python scripts/generate_large_holdout.py --skill "${skill}" --n 400 --seed 42
        done
    else
        echo "  Skipping dataset gen (SKIP_DATASET_GEN=1)"
    fi
    echo
    echo "Run: uv run python scripts/analysis/study_a_pick_n.py"
    echo "Then enter the picked values below (or set N_STAR/RATIO_STAR in env to skip)."
    prompt_if_unset N_STAR "N_STAR (one of 50, 100, 150, 250, 400)"
    prompt_if_unset RATIO_STAR "RATIO_STAR (one of 0.36, 0.50, 0.65)"
    ${STATUS_CMD}
    confirm "Proceed to Stage 5 (Study C — 12 evolve runs at the picked N*/ratio*)?"
fi

# ----------------------------------------------------------------------
# Stage 5 — Study C: 12 growth/control runs (smoke = run 1/12)
# ----------------------------------------------------------------------
if stage_active 5; then
    : "${N_STAR:?N_STAR required for Stage 5+ — set in env or via SKIP_TO_STAGE=4}"
    : "${RATIO_STAR:?RATIO_STAR required for Stage 5+ — set in env or via SKIP_TO_STAGE=4}"
    echo "--- Stage 5: Study C (12 evolve runs) ---"

    # Smoke run = nano-pdf seed=42 lenient + bap-safety=0.0. If growth_pct < 0,
    # abort — the lenient/BAP-off setup didn't produce growth and (free, slope)
    # calibration is not viable on this corpus.
    echo ">>> SMOKE: nano-pdf seed=42 (= run 1/12 of Study C)"
    ${EVOLVE} \
        --skill nano-pdf --seed 42 \
        --quality-gate lenient --bap-safety-margin 0.0 \
        --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
    echo "Inspect: jq '.growth_pct' output/nano-pdf/<latest>/gate_decision.json"
    confirm "Did the smoke run produce growth_pct > 0?"

    # Remaining 7 growth runs (3 skills × 2 seeds + 1 skill × 1 seed)
    declare -a REMAINING_GROWTH=(
        "nano-pdf 7"
        "apple-notes 42" "apple-notes 7"
        "polymarket 42" "polymarket 7"
        "huggingface-hub 42" "huggingface-hub 7"
    )
    for entry in "${REMAINING_GROWTH[@]}"; do
        read -r skill seed <<<"${entry}"
        echo ">>> GROWTH: ${skill} seed=${seed}"
        ${EVOLVE} \
            --skill "${skill}" --seed "${seed}" \
            --quality-gate lenient --bap-safety-margin 0.0 \
            --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
        ${STATUS_CMD}
        confirm "Continue Stage 5?"
    done

    # 4 control runs at default preset — confirms BAP-off is the variable
    # causing growth, not anything else in the lenient setup.
    declare -a CONTROL_PAIRS=(
        "nano-pdf 42"
        "apple-notes 42"
        "polymarket 42"
        "huggingface-hub 42"
    )
    for entry in "${CONTROL_PAIRS[@]}"; do
        read -r skill seed <<<"${entry}"
        echo ">>> CONTROL: ${skill} seed=${seed} (--quality-gate default)"
        ${EVOLVE} \
            --skill "${skill}" --seed "${seed}" \
            --quality-gate default \
            --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
    done
    ${STATUS_CMD}
fi

# ----------------------------------------------------------------------
# Stage 6 — Study B + (free, slope) analysis (no new evolve runs)
# ----------------------------------------------------------------------
if stage_active 6; then
    echo "--- Stage 6: analysis on Study C outputs ---"
    echo "Run: uv run python scripts/analysis/study_b_pick_epsilon.py"
    echo "Run: uv run python scripts/analysis/study_c_pick_curve.py"
    prompt_if_unset EPSILON_MULTIPLIER "EPSILON_MULTIPLIER (one of 0.5, 1.0, 2.0, 3.0)"
    prompt_if_unset FREE_STAR "FREE_STAR (numeric, or 'KEEP_CURRENT' if Study C verdict was INSUFFICIENT_DATA)"
    if [[ "${FREE_STAR}" != "KEEP_CURRENT" ]]; then
        prompt_if_unset SLOPE_STAR "SLOPE_STAR (numeric)"
    fi
fi

# ----------------------------------------------------------------------
# Stage 7 — Study D: 12 validation runs (current vs proposed defaults)
# ----------------------------------------------------------------------
if stage_active 7; then
    : "${N_STAR:?}" "${RATIO_STAR:?}" "${EPSILON_MULTIPLIER:?}" "${FREE_STAR:?}"
    if [[ "${FREE_STAR}" != "KEEP_CURRENT" ]]; then
        : "${SLOPE_STAR:?}"
    fi
    # max_absolute_chars=5000 (default preset) is out-of-scope for this
    # calibration but bottlenecks Stage 7: maps (6643) and linear (11195)
    # are already over the ceiling, so evolved variants get rejected on
    # absolute_char_ceiling before the (free, slope) calibration is even
    # exercised. Bump to 12000 for Stage 7 only — both arms use the same
    # value so the comparison stays clean. NOT applied to calibration
    # runs because Study C's corpus is small enough that 5000 was fine.
    STAGE7_MAX_CHARS=12000

    echo "--- Stage 7: Study D (validation runs, max_chars=${STAGE7_MAX_CHARS}) ---"
    confirm "Proceed to Stage 7?"
    for skill in "${SKILLS_VALIDATION[@]}"; do
        for seed in 42 7; do
            # Fresh ε per run since it scales with n_val, which the runtime
            # computes — but Stage 7 uses the picked multiplier consistently.
            # We pass the multiplier through --knee-point-epsilon as a hint
            # only; the runtime falls back to multiplier/n_val.
            echo ">>> VALIDATION current: ${skill} seed=${seed}"
            ${EVOLVE} \
                --skill "${skill}" --seed "${seed}" \
                --quality-gate default \
                --max-absolute-chars "${STAGE7_MAX_CHARS}"
            echo ">>> VALIDATION proposed: ${skill} seed=${seed}"
            if [[ "${FREE_STAR}" == "KEEP_CURRENT" ]]; then
                ${EVOLVE} \
                    --skill "${skill}" --seed "${seed}" \
                    --quality-gate default \
                    --max-absolute-chars "${STAGE7_MAX_CHARS}" \
                    --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
            else
                ${EVOLVE} \
                    --skill "${skill}" --seed "${seed}" \
                    --quality-gate default \
                    --max-absolute-chars "${STAGE7_MAX_CHARS}" \
                    --growth-free-threshold "${FREE_STAR}" \
                    --growth-quality-slope "${SLOPE_STAR}" \
                    --eval-dataset-size "${N_STAR}" --holdout-ratio "${RATIO_STAR}"
            fi
            ${STATUS_CMD}
            confirm "Continue Stage 7?"
        done
    done
fi

echo
echo "=== Campaign runs complete ==="
echo "Run: uv run python scripts/analysis/study_d_compare.py --since ${CAMPAIGN_START_TS}"
${STATUS_CMD}
