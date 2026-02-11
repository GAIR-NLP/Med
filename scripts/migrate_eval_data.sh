#!/bin/bash
#
# Bash wrapper for migrating evaluation data
# Provides convenient presets for common migration tasks
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/migrate_eval_data.py"

# Predefined benchmark groups
REASONING_BENCHMARKS="charxiv2rq,mathvision,mathvista,mme,mmmu,mmmupro"
PERCEPTION_BENCHMARKS="vstar,hrbench4k,hrbench8k,visualprobeasy,visualprobmedium,visualprobhard"
# PERCEPTION_BENCHMARKS="vstar,hrbench4k,hrbench8k,visualprobeasy,visualprobmedium,visualprobhard,zerobench"
ALL_BENCHMARKS="$REASONING_BENCHMARKS,$PERCEPTION_BENCHMARKS"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] EXP_PATH [EXP_PATH ...]

Migrate evaluation data from verl_vision to Med/evals.

Arguments:
  EXP_PATH           One or more experiment paths in format:
                     - 'group' to migrate entire group (e.g., qwen3vl_instruct_75_50)
                     - 'group/exp_name' for single experiment
                       (e.g., qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1)

Options:
  --reasoning        Migrate only reasoning benchmarks
  --perception       Migrate only perception benchmarks
  --benchmarks LIST  Comma-separated benchmark names
  --steps RANGE      Step range (e.g., 10-100 or 10,20,30)
  --override         Override existing files
  --dry-run          Preview without migrating
  -v, --verbose      Verbose output
  -h, --help         Show this help message

Examples:
  # Migrate all experiments in a group
  $0 qwen3vl_instruct_75_50

  # Migrate specific experiment
  $0 qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1

  # Migrate multiple experiments at once
  $0 qwen3vl_instruct_75_50/exp1 qwen3vl_instruct_75_50/exp2 qwen25vl_instruct_75_50/exp3

  # Migrate multiple experiments with options
  $0 qwen3vl_instruct_75_50/exp1 qwen25vl_instruct_75_50/exp2 --reasoning --steps 10-100

  # Migrate only reasoning benchmarks
  $0 --reasoning qwen3vl_instruct_75_50

  # Migrate specific benchmarks and steps
  $0 --benchmarks charxiv2rq,mathvision --steps 10-50 qwen3vl_instruct_75_50/qwen3vl_natural_0.75_toolcall_0.5_cons_0.1

  # Preview migration
  $0 --dry-run qwen3vl_instruct_75_50

Benchmark Categories:
  Reasoning:  $REASONING_BENCHMARKS
  Perception: $PERCEPTION_BENCHMARKS

EOF
    exit 1
}

# Parse arguments
BENCHMARKS=""
STEPS=""
OVERRIDE=""
DRY_RUN=""
VERBOSE=""
EXP_PATHS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --reasoning)
            BENCHMARKS="$REASONING_BENCHMARKS"
            shift
            ;;
        --perception)
            BENCHMARKS="$PERCEPTION_BENCHMARKS"
            shift
            ;;
        --benchmarks)
            BENCHMARKS="$2"
            shift 2
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --override)
            OVERRIDE="--override"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            # Collect all non-option arguments as experiment paths
            EXP_PATHS+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [[ ${#EXP_PATHS[@]} -eq 0 ]]; then
    echo "Error: At least one EXP_PATH is required"
    usage
fi

# Build base command with options
BASE_CMD="python3 $PYTHON_SCRIPT"

if [[ -n "$BENCHMARKS" ]]; then
    BASE_CMD="$BASE_CMD --benchmarks $BENCHMARKS"
fi

if [[ -n "$STEPS" ]]; then
    BASE_CMD="$BASE_CMD --steps $STEPS"
fi

if [[ -n "$OVERRIDE" ]]; then
    BASE_CMD="$BASE_CMD $OVERRIDE"
fi

if [[ -n "$DRY_RUN" ]]; then
    BASE_CMD="$BASE_CMD $DRY_RUN"
fi

if [[ -n "$VERBOSE" ]]; then
    BASE_CMD="$BASE_CMD $VERBOSE"
fi

# Process each experiment path
TOTAL_COUNT=${#EXP_PATHS[@]}
CURRENT=0

for EXP_PATH in "${EXP_PATHS[@]}"; do
    CURRENT=$((CURRENT + 1))

    if [[ $TOTAL_COUNT -gt 1 ]]; then
        echo -e "${GREEN}[$CURRENT/$TOTAL_COUNT]${NC} Processing: ${YELLOW}$EXP_PATH${NC}"
    fi

    CMD="$BASE_CMD $EXP_PATH"

    if [[ $TOTAL_COUNT -eq 1 ]] || [[ -n "$VERBOSE" ]]; then
        echo -e "${YELLOW}Running:${NC} $CMD"
    fi
    echo ""

    # Execute
    eval $CMD

    if [[ $TOTAL_COUNT -gt 1 ]]; then
        echo ""
        echo "---"
        echo ""
    fi
done

# Print final summary if processing multiple experiments
if [[ $TOTAL_COUNT -gt 1 ]]; then
    echo -e "${GREEN}Completed processing $TOTAL_COUNT experiments${NC}"
fi
