#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_merge_data.sh [DATASET_ROOT] [PATTERN] [OUTPUT_NAME]
# Example:
#   ./run_merge_data.sh $(pwd)/data/multi_object_open/lerobot 7221 multi_object_open_7221

DATASET_ROOT_DEFAULT="$(pwd)/data/multi_object_open/lerobot"
PATTERN_DEFAULT="7221"
OUTPUT_DEFAULT="multi_object_open_7221_merged"

DATASET_ROOT="${1:-$DATASET_ROOT_DEFAULT}"
PATTERN="${2:-$PATTERN_DEFAULT}"
OUTPUT_NAME="${3:-$OUTPUT_DEFAULT}"
DRY_RUN=0
FORCE=0
# Parse optional flags from remaining args (supports --dry-run/-n and --force/-f)
for arg in "${@:4}"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --force|-f) FORCE=1 ;;
    esac
done
# Collect matching repo directories (only immediate children), skipping the intended output folder and any directories that include 'merge' in their name
repos=()
for d in "$DATASET_ROOT"/*; do
    base=$(basename "$d")
    if [[ -d "$d" && "$base" == *"$PATTERN"* ]]; then
        # Skip if this is exactly the output folder itself or contains 'merge' (to avoid including merged outputs)
        if [[ "$base" == "$OUTPUT_NAME" ]] || [[ "$base" == *"merge"* ]] || [[ "$base" == *"merged"* ]]; then
            echo "Skipping $d (output folder or contains 'merge')"
            continue
        fi
        repos+=("$d")
    fi
done

if [ ${#repos[@]} -eq 0 ]; then
    echo "No matching repositories found in '$DATASET_ROOT' with pattern '$PATTERN' after filtering out '$OUTPUT_NAME' and any 'merge' folders"
    exit 1
fi

# Build repo_ids string as a single-line JSON array (double-quoted) to avoid embedded newlines
repo_ids_str="["
for ((i=0;i<${#repos[@]};i++)); do
    # escape any double quotes in paths
    esc=$(printf '%s' "${repos[$i]}" | sed 's/"/\\"/g')
    if [ $i -gt 0 ]; then
        repo_ids_str+=","
    fi
    repo_ids_str+="\"${esc}\""
done
repo_ids_str+="]"
# Show constructed repo_ids_str for debugging
echo "repo_ids_str: $repo_ids_str"

# Show what we will run
echo "Merging the following repositories into '$OUTPUT_NAME':"
for r in "${repos[@]}"; do
    echo "  - $r"
done

echo
# Ensure 'lerobot' module is importable; if not, add local third_party path to PYTHONPATH
if ! python -c "import lerobot" &>/dev/null; then
    export PYTHONPATH="$(pwd)/third_party/lerobot/src${PYTHONPATH:+:}${PYTHONPATH:-}"
    echo "Prepended third_party/lerobot/src to PYTHONPATH"
fi

# Run merge
python -m lerobot.scripts.lerobot_edit_dataset \
    --repo_id "$DATASET_ROOT/$OUTPUT_NAME" \
    --operation.type merge \
    --operation.repo_ids "$repo_ids_str"

echo "Done. Merged into: $DATASET_ROOT/$OUTPUT_NAME"
