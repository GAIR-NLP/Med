#!/bin/bash
#
# Bash wrapper for migrating baseline evaluation data
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/migrate_baseline_data.py"

# Predefined benchmark groups
REASONING_BENCHMARKS="charxiv2rq,mathvision,mathvista,mme,mmmu,mmmupro"
PERCEPTION_BENCHMARKS="vstar,hrbench4k,hrbench8k,visualprobeasy,visualprobmedium,visualprobhard"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS] MODEL_NAME [MODEL_NAME ...]

Migrate baseline evaluation data from verl_vision to Med/evals/baseline.

Arguments:
  MODEL_NAME         One or more baseline model names
                     (e.g., Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct)

Options:
  --reasoning        Migrate only reasoning benchmarks
  --perception       Migrate only perception benchmarks
  --benchmarks LIST  Comma-separated benchmark names
  --override         Override existing files
  --dry-run          Preview without migrating
  -v, --verbose      Verbose output
  -h, --help         Show this help message

Examples:
  # Migrate single baseline model
  $0 Qwen2.5-VL-7B-Instruct

  # Migrate both baseline models
  $0 Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct

  # Migrate with specific benchmarks
  $0 Qwen2.5-VL-7B-Instruct --benchmarks charxiv2rq,mathvision

  # Migrate only reasoning benchmarks
  $0 Qwen2.5-VL-7B-Instruct Qwen3-VL-8B-Instruct --reasoning

  # Preview migration
  $0 Qwen2.5-VL-7B-Instruct --dry-run

Benchmark Categories:
  Reasoning:  $REASONING_BENCHMARKS
  Perception: $PERCEPTION_BENCHMARKS

EOF
    exit 1
}

# Parse arguments
BENCHMARKS=""
OVERRIDE=""
DRY_RUN=""
VERBOSE=""
MODEL_NAMES=()

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
            # Collect all non-option arguments as model names
            MODEL_NAMES+=("$1")
            shift
            ;;
    esac
done

# Validate arguments
if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    echo "Error: At least one MODEL_NAME is required"
    usage
fi

# Build command
CMD="python3 $PYTHON_SCRIPT ${MODEL_NAMES[@]}"

if [[ -n "$BENCHMARKS" ]]; then
    CMD="$CMD --benchmarks $BENCHMARKS"
fi

if [[ -n "$OVERRIDE" ]]; then
    CMD="$CMD $OVERRIDE"
fi

if [[ -n "$DRY_RUN" ]]; then
    CMD="$CMD $DRY_RUN"
fi

if [[ -n "$VERBOSE" ]]; then
    CMD="$CMD $VERBOSE"
fi

# Print command
echo -e "${YELLOW}Running:${NC} $CMD"
echo ""

# Execute
eval $CMD
