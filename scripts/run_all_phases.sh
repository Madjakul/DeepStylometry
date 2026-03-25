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
# --resume      Skip training jobs whose last.ckpt already exists.
#               Test jobs still run if the checkpoint is present.
#
# --train-only  Submit only training jobs; skip all test submissions.
#               Use this to resubmit failed training runs without triggering
#               redundant tests while other jobs are still in the queue.
#
# --test-only   Submit only test jobs (skips all training).  Combine with
#               --resume to avoid submitting tests for missing checkpoints:
#                 ./scripts/run_all_phases.sh --phase fixed_patches --test-only
#               Each test job is submitted with no dependency (runs immediately).
#
# --dry-run     Echo sbatch commands without submitting.
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
TRAIN_ONLY=false
TEST_ONLY=false

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
        --train-only)
            TRAIN_ONLY=true; shift ;;
        --test-only)
            TEST_ONLY=true; shift ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [--phase baselines|fixed_patches|tune|learned|appendix|all] [--resume] [--train-only] [--test-only] [--dry-run]" >&2
            exit 1 ;;
    esac
done

if $TRAIN_ONLY && $TEST_ONLY; then
    echo "Error: --train-only and --test-only are mutually exclusive." >&2
    exit 1
fi

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
        echo "[DRY-RUN] sbatch ${dep:+$dep }$*  # $desc" >&2
        echo "0"   # fake job id (stdout only, captured by caller)
        return
    fi
    local jid
    # Pass --output with absolute path so SLURM never fails to find logs/
    # regardless of the submit working directory.
    jid=$(sbatch --parsable \
        --output="$PROJECT_ROOT/logs/%x-%j-%t.log" \
        ${dep:+$dep} "$@")
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

# Return true (0) if training should be skipped:
#   - --resume and checkpoint already exists, OR
#   - --test-only mode
skip_train() {
    local ckpt="$1"
    $TEST_ONLY && return 0
    $RESUME && [[ -f "$ckpt" ]]
}

# Kept for backward compat (used in resume messages below)
ckpt_exists() {
    local ckpt="$1"
    $RESUME && [[ -f "$ckpt" ]]
}

# Run name mirrors train_utils.py logic (lowercase, / → -)
# For pli: run_name_std pli <skip> <patch_method> <patch_size> [<patch_compression>]
# patch_compression is appended to patch_tag only when provided and not "mean".
run_name_std() {
    local pool="$1" skip="$2"
    local base="answerdotai-modernbert-base__halvest"
    if [[ "$pool" == "pli" ]]; then
        local pm="$3" ps="$4" compression="${5:-mean}"
        local patch_tag="${pm}-n${ps}"
        if [[ "$compression" != "mean" ]]; then
            patch_tag="${patch_tag}-${compression}"
        fi
        echo "${base}__pooling-pli-${patch_tag}__skip_list-${skip}"
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

    # Skip entirely when --train-only is set
    $TRAIN_ONLY && return 0

    local dep
    # --test-only: no upstream training job, run immediately
    if $TEST_ONLY; then
        dep=""
    else
        dep=$(dep_afterany "$train_jid")
    fi

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
    if ! skip_train "$CKPT_MEAN"; then
        JID_TRAIN_MEAN=$(submit "train B2 mean" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_mean.yml")
    else
        echo "[SKIP] B2 mean training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_mean" "$CKPT_MEAN" "$JID_TRAIN_MEAN" > /dev/null

    # B3 — LateInteraction
    echo "=== B3: LateInteraction baseline ==="
    CKPT_LI=$CHECKPOINT_DIR/$(run_name_std li true)/last.ckpt
    JID_TRAIN_LI=""
    if ! skip_train "$CKPT_LI"; then
        JID_TRAIN_LI=$(submit "train B3 li" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train.yml")
    else
        echo "[SKIP] B3 li training skipped (--test-only or checkpoint exists)." >&2
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
    if ! skip_train "$CKPT_WS"; then
        JID_TRAIN_WS=$(submit "train P1 pli-whitespace" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_whitespace.yml")
    else
        echo "[SKIP] P1 whitespace training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_whitespace" "$CKPT_WS" "$JID_TRAIN_WS" > /dev/null

    # P2 — PLI wholeword
    echo "=== P2: PLI wholeword ==="
    CKPT_WW=$CHECKPOINT_DIR/$(run_name_std pli true wholeword 1)/last.ckpt
    JID_TRAIN_WW=""
    if ! skip_train "$CKPT_WW"; then
        JID_TRAIN_WW=$(submit "train P2 pli-wholeword" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_wholeword.yml")
    else
        echo "[SKIP] P2 wholeword training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_wholeword" "$CKPT_WW" "$JID_TRAIN_WW" > /dev/null

    # P3 — PLI ngram variants.
    # Training is chained 2-by-2 to avoid hitting the per-account GPU quota:
    #   chain A: ngram2 ──afterany──▶ ngram4
    #   chain B: ngram3 ──afterany──▶ ngram5
    # Tests are submitted with afterany on their own training job (unchanged).
    echo "=== P3: PLI ngram-{2,3,4,5} (chains: 2→4 and 3→5) ==="

    # -- ngram2 (no upstream dependency) --
    CKPT_N2=$CHECKPOINT_DIR/$(run_name_std pli true ngram 2)/last.ckpt
    JID_N2=""
    if ! skip_train "$CKPT_N2"; then
        JID_N2=$(submit "train P3 pli-ngram2" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_ngram2.yml")
    else
        echo "[SKIP] P3 ngram-2 training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_ngram2" "$CKPT_N2" "$JID_N2" > /dev/null

    # -- ngram3 (no upstream dependency) --
    CKPT_N3=$CHECKPOINT_DIR/$(run_name_std pli true ngram 3)/last.ckpt
    JID_N3=""
    if ! skip_train "$CKPT_N3"; then
        JID_N3=$(submit "train P3 pli-ngram3" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_ngram3.yml")
    else
        echo "[SKIP] P3 ngram-3 training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_ngram3" "$CKPT_N3" "$JID_N3" > /dev/null

    # -- ngram4 (starts after ngram2 finishes) --
    CKPT_N4=$CHECKPOINT_DIR/$(run_name_std pli true ngram 4)/last.ckpt
    JID_N4=""
    if ! skip_train "$CKPT_N4"; then
        JID_N4=$(submit "train P3 pli-ngram4" "$(dep_afterany "$JID_N2")" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_ngram4.yml")
    else
        echo "[SKIP] P3 ngram-4 training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_ngram4" "$CKPT_N4" "$JID_N4" > /dev/null

    # -- ngram5 (starts after ngram3 finishes) --
    CKPT_N5=$CHECKPOINT_DIR/$(run_name_std pli true ngram 5)/last.ckpt
    JID_N5=""
    if ! skip_train "$CKPT_N5"; then
        JID_N5=$(submit "train P3 pli-ngram5" "$(dep_afterany "$JID_N3")" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_ngram5.yml")
    else
        echo "[SKIP] P3 ngram-5 training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_ngram5" "$CKPT_N5" "$JID_N5" > /dev/null

    # Expose last submitted training JID for tune phase
    LAST_P3_JID="${JID_N5:-${JID_N4:-${JID_N3:-$JID_N2}}}"
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
        "$PROJECT_ROOT/scripts/tune.sbatch" "$CONFIGS/train_pli_learned.yml")
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
    if ! skip_train "$CKPT_LEARNED"; then
        JID_TRAIN_LEARNED=$(submit "train P4 pli-learned" "$dep" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_learned.yml")
    else
        echo "[SKIP] P4 learned training skipped (--test-only or checkpoint exists)." >&2
    fi
    submit_tests "test_pli_learned" "$CKPT_LEARNED" "$JID_TRAIN_LEARNED" > /dev/null
}

# ---------------------------------------------------------------------------
# Phase: appendix  (PLI learned + cross-attention compression)
# ---------------------------------------------------------------------------
run_appendix() {
    echo "=== Appendix: PLI learned + cross-attention compression ==="
    CKPT_XATTN=$CHECKPOINT_DIR/$(run_name_std pli true learned 3 cross_attention)/last.ckpt
    JID_TRAIN_XATTN=""
    if ! skip_train "$CKPT_XATTN"; then
        JID_TRAIN_XATTN=$(submit "train appendix pli-xattn" "" \
            "$PROJECT_ROOT/scripts/train.sbatch" "$CONFIGS/train_pli_learned_xattn.yml")
    else
        echo "[SKIP] Appendix xattn training skipped (--test-only or checkpoint exists)." >&2
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
