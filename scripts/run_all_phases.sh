#!/bin/bash
# scripts/run_all_phases.sh
#
# Master orchestration script for the full PLI experiment pipeline.
# Submits SLURM jobs with dependency chaining so each phase runs only
# after its prerequisites succeed.
#
# Usage:
#   ./scripts/run_all_phases.sh [--phase PHASE] [--resume] [--dry-run]
#
# --phase PHASE   Run only the named phase group (default: all):
#     baselines      — B2 (mean) + B3 (LI) only
#     fixed_patches  — P1 (whitespace) + P2 (wholeword) + P3 (ngram-2..5)
#     tune           — Optuna hyperparameter search for PLI
#     learned        — P4 (PLI learned boundaries, mean compression)
#     appendix       — Appendix: P4 with cross-attention compression
#     all            — everything (default)
#
# --resume   Skip phases whose last.ckpt already exists on disk.
#            Test jobs still run if the checkpoint is present.
#
# --dry-run  Echo sbatch commands without submitting.
#
# SLURM dependency notes:
#   --dependency=afterok  : downstream job only starts on upstream success.
#   --dependency=afterany : downstream job starts regardless of upstream exit
#                           status (used for test jobs, so they always attempt
#                           even if the training run produced only a partial
#                           checkpoint).
#
# Clarification: sbatch submits jobs and returns immediately.  The shell
# script itself finishes in under a second; SLURM manages the dependency
# chain server-side.  Closing the terminal or logging out does NOT cancel
# queued jobs.

set -euo pipefail

PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
CONFIGS=$PROJECT_ROOT/configs
CHECKPOINT_DIR=${CHECKPOINT_DIR:-$PROJECT_ROOT/tmp}
PROCESSED_DS_DIR=${PROCESSED_DS_DIR:-${WORK_DIR:-$HOME}/Datasets/deep-stylometry/answerdotai-modernbert-base/no-padding/}

PHASE="all"
DRY_RUN=false
RESUME=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase)
            PHASE="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --resume)
            RESUME=true; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--phase baselines|fixed_patches|tune|learned|appendix|all] [--resume] [--dry-run]" >&2
            exit 1 ;;
    esac
done

if $DRY_RUN; then
    echo "[DRY-RUN] No jobs will be submitted."
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

submit() {
    # submit <description> <dep_flag_or_empty> <sbatch_script> [extra_args...]
    local desc="$1"; shift
    local dep="$1"; shift   # e.g. "--dependency=afterok:123" or ""
    if $DRY_RUN; then
        echo "[DRY-RUN] sbatch ${dep:+$dep }$*  # $desc"
        echo "0"   # fake job id
        return
    fi
    local jid
    jid=$(sbatch --parsable ${dep:+$dep} "$@")
    echo "$jid"
    echo "[SUBMITTED] $desc → job $jid" >&2
}

dep_afterok() {
    local jid="$1"
    if [[ -z "$jid" || "$jid" == "0" ]]; then echo ""; else echo "--dependency=afterok:$jid"; fi
}

dep_afterany() {
    local jid="$1"
    if [[ -z "$jid" || "$jid" == "0" ]]; then echo ""; else echo "--dependency=afterany:$jid"; fi
}

# Return true (0) if the checkpoint already exists and --resume is set
ckpt_exists() {
    local ckpt="$1"
    $RESUME && [[ -f "$ckpt" ]]
}

# Run name mirrors train_utils.py logic (lowercase, / → -)
run_name_std() {
    local pool="$1" skip="$2"
    local base="answerdotai-modernbert-base__halvest"
    if [[ "$pool" == "pli" ]]; then
        local pm="$3" ps="$4"
        echo "${base}__pooling-pli-${pm}-n${ps}__skip_list-${skip}"
    else
        echo "${base}__pooling-${pool}__skip_list-${skip}"
    fi
}

# ---------------------------------------------------------------------------
# Test job helper — submits tests on 4 subsets after a training job
# Test jobs use afterany so they still run even if training produced only a
# partial checkpoint (fault tolerance).
# ---------------------------------------------------------------------------
submit_tests() {
    local cfg_prefix="$1"   # e.g. test_mean, test_pli_ngram3
    local ckpt_path="$2"    # path to last.ckpt
    local train_jid="$3"    # job id to depend on (may be "" if --resume skipped train)

    local dep
    dep=$(dep_afterany "$train_jid")

    local jids=()
    for subset in base-2 base-4 base-8 unrestricted; do
        local jid
        jid=$(submit \
            "test ${cfg_prefix} subset=${subset}" \
            "$dep" \
            "$PROJECT_ROOT/scripts/test.sbatch" \
            "$CONFIGS/${cfg_prefix}.yml" \
            "$ckpt_path" \
            -- \
            --test_subset "$subset")
        jids+=("$jid")
    done

    # Zero-shot SE eval
    local jid_se
    jid_se=$(submit \
        "test ${cfg_prefix} se (zero-shot)" \
        "$dep" \
        "$PROJECT_ROOT/scripts/test.sbatch" \
        "$CONFIGS/${cfg_prefix}.yml" \
        "$ckpt_path" \
        -- \
        --ds_name se)
    jids+=("$jid_se")

    local IFS=","
    echo "${jids[*]}"
}

# ---------------------------------------------------------------------------
# Phase: baselines  (B2 + B3)
# ---------------------------------------------------------------------------
run_baselines() {
    # B2 — Mean pooling
    echo "=== B2: MeanInteraction baseline ==="
    CKPT_MEAN=$CHECKPOINT_DIR/$(run_name_std mean false)/last.ckpt
    JID_TRAIN_MEAN=""
    if ! ckpt_exists "$CKPT_MEAN"; then
        JID_TRAIN_MEAN=$(submit "train B2 mean" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_mean.yml")
    else
        echo "[RESUME] Checkpoint found for B2 mean, skipping training." >&2
    fi
    submit_tests "test_mean" "$CKPT_MEAN" "$JID_TRAIN_MEAN" > /dev/null

    # B3 — LateInteraction
    echo "=== B3: LateInteraction baseline ==="
    CKPT_LI=$CHECKPOINT_DIR/$(run_name_std li true)/last.ckpt
    JID_TRAIN_LI=""
    if ! ckpt_exists "$CKPT_LI"; then
        JID_TRAIN_LI=$(submit "train B3 li" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train.yml")
    else
        echo "[RESUME] Checkpoint found for B3 li, skipping training." >&2
    fi
    submit_tests "test" "$CKPT_LI" "$JID_TRAIN_LI" > /dev/null
}

# ---------------------------------------------------------------------------
# Phase: fixed_patches  (P1 + P2 + P3)
# ---------------------------------------------------------------------------
run_fixed_patches() {
    # P1 — PLI whitespace
    echo "=== P1: PLI whitespace ==="
    CKPT_WS=$CHECKPOINT_DIR/$(run_name_std pli true whitespace 1)/last.ckpt
    JID_TRAIN_WS=""
    if ! ckpt_exists "$CKPT_WS"; then
        JID_TRAIN_WS=$(submit "train P1 pli-whitespace" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_whitespace.yml")
    else
        echo "[RESUME] Checkpoint found for P1 whitespace, skipping training." >&2
    fi
    submit_tests "test_pli_whitespace" "$CKPT_WS" "$JID_TRAIN_WS" > /dev/null

    # P2 — PLI wholeword
    echo "=== P2: PLI wholeword ==="
    CKPT_WW=$CHECKPOINT_DIR/$(run_name_std pli true wholeword 1)/last.ckpt
    JID_TRAIN_WW=""
    if ! ckpt_exists "$CKPT_WW"; then
        JID_TRAIN_WW=$(submit "train P2 pli-wholeword" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_wholeword.yml")
    else
        echo "[RESUME] Checkpoint found for P2 wholeword, skipping training." >&2
    fi
    submit_tests "test_pli_wholeword" "$CKPT_WW" "$JID_TRAIN_WW" > /dev/null

    # P3 — PLI ngram variants (all submitted in parallel; last jid exposed for Tune)
    echo "=== P3: PLI ngram-{2,3,4,5} ==="
    LAST_P3_JID=""
    for N in 2 3 4 5; do
        CKPT_NG=$CHECKPOINT_DIR/$(run_name_std pli true ngram $N)/last.ckpt
        JID=""
        if ! ckpt_exists "$CKPT_NG"; then
            JID=$(submit "train P3 pli-ngram${N}" "" \
                "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_ngram${N}.yml")
        else
            echo "[RESUME] Checkpoint found for P3 ngram-${N}, skipping training." >&2
        fi
        submit_tests "test_pli_ngram${N}" "$CKPT_NG" "$JID" > /dev/null
        LAST_P3_JID="$JID"
    done
    # Export so tune phase can pick it up
    export LAST_P3_JID
}

# ---------------------------------------------------------------------------
# Phase: tune  (Optuna HPT search)
# ---------------------------------------------------------------------------
run_tune() {
    local upstream_jid="${LAST_P3_JID:-}"
    local dep
    dep=$(dep_afterok "$upstream_jid")
    echo "=== Tune: Optuna PLI hyperparameter search ==="
    JID_TUNE=$(submit "tune PLI optuna" "$dep" \
        "$PROJECT_ROOT/scripts/tune.sbatch" "$CONFIGS/train_pli_ngram3.yml")
    export JID_TUNE
}

# ---------------------------------------------------------------------------
# Phase: learned  (P4 — PLI learned boundaries)
# ---------------------------------------------------------------------------
run_learned() {
    local upstream_jid="${JID_TUNE:-}"
    local dep
    dep=$(dep_afterok "$upstream_jid")
    echo "=== P4: PLI learned boundaries ==="
    CKPT_LEARNED=$CHECKPOINT_DIR/$(run_name_std pli true learned 3)/last.ckpt
    JID_TRAIN_LEARNED=""
    if ! ckpt_exists "$CKPT_LEARNED"; then
        JID_TRAIN_LEARNED=$(submit "train P4 pli-learned" "$dep" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_learned.yml")
    else
        echo "[RESUME] Checkpoint found for P4 learned, skipping training." >&2
    fi
    submit_tests "test_pli_learned" "$CKPT_LEARNED" "$JID_TRAIN_LEARNED" > /dev/null
}

# ---------------------------------------------------------------------------
# Phase: appendix  (PLI learned + cross-attention compression)
# ---------------------------------------------------------------------------
run_appendix() {
    echo "=== Appendix: PLI learned + cross-attention compression ==="
    CKPT_XATTN=$CHECKPOINT_DIR/$(run_name_std pli true learned-xattn 3)/last.ckpt
    JID_TRAIN_XATTN=""
    if ! ckpt_exists "$CKPT_XATTN"; then
        JID_TRAIN_XATTN=$(submit "train appendix pli-xattn" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_learned_xattn.yml")
    else
        echo "[RESUME] Checkpoint found for appendix xattn, skipping training." >&2
    fi
    submit_tests "test_pli_learned_xattn" "$CKPT_XATTN" "$JID_TRAIN_XATTN" > /dev/null
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
LAST_P3_JID=""
JID_TUNE=""

# Ensure the logs directory exists before any sbatch call.
# SLURM resolves #SBATCH --output=logs/... at job start; if the directory is
# missing the job is immediately killed before executing any code.
mkdir -p "$PROJECT_ROOT/logs"

case "$PHASE" in
    baselines)
        run_baselines ;;
    fixed_patches)
        run_fixed_patches ;;
    tune)
        run_tune ;;
    learned)
        run_learned ;;
    appendix)
        run_appendix ;;
    all)
        run_baselines
        run_fixed_patches
        run_tune
        run_learned
        run_appendix ;;
    *)
        echo "Unknown phase: $PHASE" >&2
        echo "Valid phases: baselines, fixed_patches, tune, learned, appendix, all" >&2
        exit 1 ;;
esac

echo ""
echo "=== All jobs submitted (phase: $PHASE) ==="
if ! $DRY_RUN; then
    echo "Monitor with: squeue -u \$USER"
fi
