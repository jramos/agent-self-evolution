#!/usr/bin/env bash
# BAP max_growth calibration runbook (Stage 2 of the post-deploy-gate
# calibration plan).
#
# Sweeps --bap-max-growth ∈ {0.05, 0.10, 0.15, 0.20, 0.30} on linear and
# notion (both >5000 chars baseline; rely on the absolute-char-ceiling
# scaling fix to not bottleneck on size). 2 seeds per value × 2 skills
# × 5 values = 20 runs. Each run ~$3-5; total budget ~$60-100.
#
# Pre-reqs:
#   - feat/bap-decouple-and-fixes merged (or this branch built atop it).
#   - SKILL_SOURCES_HERMES_REPO set so --skill resolves.
#   - OPENAI_API_KEY populated.
#   - ulimit -n 65536 (asyncio FD safety, same lesson as previous campaign).
#
# Usage:
#   ulimit -n 65536
#   bash scripts/run_bap_campaign.sh
#
# Resume hints (set on the same line as the bash invocation):
#   CAMPAIGN_START_TS=20260510_120000 SKIP_SMOKE=1 \
#       bash scripts/run_bap_campaign.sh
#   SKIP_TO_INDEX=8 bash scripts/run_bap_campaign.sh
#       # ...skip the first 8 sweep entries (already done).

set -euo pipefail

CAMPAIGN_START_TS="${CAMPAIGN_START_TS:-$(date +%Y%m%d_%H%M%S)}"
CAP_USD=100
STATUS_CMD="uv run python scripts/campaign_status.py --since ${CAMPAIGN_START_TS} --cap ${CAP_USD}"
EVOLVE="uv run python -m evolution.skills.evolve_skill --iterations 10 --evaluate-band-on-holdout --quality-gate default --eval-dataset-size 250 --holdout-ratio 0.65"

SKILLS=(linear notion)
MAX_GROWTHS=(0.05 0.10 0.15 0.20 0.30)
SEEDS=(42 7)
SKIP_TO_INDEX="${SKIP_TO_INDEX:-0}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"

confirm() {
    local prompt="$1"
    read -r -p "${prompt} [y/N] " answer
    case "${answer}" in
        y|Y|yes|YES) return 0 ;;
        *) echo "Aborted."; exit 1 ;;
    esac
}

echo "=== BAP max_growth calibration ==="
echo "Campaign start: ${CAMPAIGN_START_TS}"
echo "Cost cap:       \$${CAP_USD}"
echo "Skills:         ${SKILLS[*]}"
echo "Sweep:          ${MAX_GROWTHS[*]}"
echo "Seeds:          ${SEEDS[*]}"
echo "ulimit -n:      $(ulimit -n)"
echo "SKIP_SMOKE:     ${SKIP_SMOKE}"
echo "SKIP_TO_INDEX:  ${SKIP_TO_INDEX}"
echo

# ----------------------------------------------------------------------
# Pre-flight smoke: verify the absolute_char_ceiling scaling fix landed.
# linear (baseline ~11185 chars) at --bap-max-growth 0.20 should NOT
# auto-reject on absolute size. Without the scaling fix, every run
# would bounce off the static 5000 ceiling before the calibration data
# is even produced.
# ----------------------------------------------------------------------
if [[ "${SKIP_SMOKE}" != "1" ]]; then
    echo "--- Pre-flight smoke ---"
    confirm "Run a single linear evolution at --bap-max-growth 0.20 to verify the ceiling-scaling fix is live (~$5)?"
    ${EVOLVE} \
        --skill linear --seed 42 \
        --bap-max-growth 0.20

    # Find the latest linear run and verify the effective ceiling field.
    latest_dir=$(ls -1dt output/linear/2026*/ 2>/dev/null | head -1)
    effective=$(jq -r '.effective_absolute_char_ceiling // "MISSING"' "${latest_dir}gate_decision.json")
    static=$(jq -r '.absolute_char_ceiling // "MISSING"' "${latest_dir}gate_decision.json")
    echo
    echo "  Smoke check: static=${static}, effective=${effective}"
    if [[ "${effective}" == "MISSING" ]] || [[ "${effective}" == "${static}" && "${static}" == "5000" ]]; then
        echo "  ✗ effective_absolute_char_ceiling missing or still at static 5000 floor."
        echo "    The ceiling-scaling fix did not land. Aborting before spending campaign budget."
        exit 1
    fi
    echo "  ✓ Ceiling scaled correctly. Proceeding with the sweep."
    confirm "Continue to the full sweep?"
else
    echo "Skipping pre-flight smoke (SKIP_SMOKE=1)."
fi

# ----------------------------------------------------------------------
# Sweep: 2 skills × 5 max_growth values × 2 seeds = 20 runs.
# Iterate skill-major so all linear runs land before notion — keeps
# the on-disk corpus organized for incremental analysis.
# ----------------------------------------------------------------------
declare -a PLAN=()
for skill in "${SKILLS[@]}"; do
    for mg in "${MAX_GROWTHS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            PLAN+=("${skill} ${mg} ${seed}")
        done
    done
done
echo
echo "Total sweep entries: ${#PLAN[@]} (skipping first ${SKIP_TO_INDEX})"
echo

idx=0
for entry in "${PLAN[@]}"; do
    if (( idx < SKIP_TO_INDEX )); then
        idx=$((idx + 1))
        continue
    fi
    read -r skill mg seed <<<"${entry}"
    echo
    echo ">>> [${idx}/${#PLAN[@]}] ${skill} max_growth=${mg} seed=${seed}"
    ${EVOLVE} \
        --skill "${skill}" --seed "${seed}" \
        --bap-max-growth "${mg}"
    ${STATUS_CMD}
    idx=$((idx + 1))
    # Prompt every 4 runs (= 1 max_growth × 2 seeds × 2 skills, or any
    # granularity that gives the user a stop point without prompting
    # after every run).
    if (( idx % 4 == 0 && idx < ${#PLAN[@]} )); then
        confirm "Continue sweep at index ${idx}/${#PLAN[@]}?"
    fi
done

echo
echo "=== Sweep complete ==="
${STATUS_CMD}
echo
echo "Run the analysis:"
echo "  uv run python scripts/analysis/bap_max_growth_pick.py --since ${CAMPAIGN_START_TS}"
