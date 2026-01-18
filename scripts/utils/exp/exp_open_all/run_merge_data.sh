#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_merge_data.sh [DATASET_ROOT] [OUTPUT_NAME] [--dry-run]
# Example:
#   ./run_merge_data.sh $(pwd)/data/multi_object_open/lerobot multi_object_open_all --dry-run

DATASET_ROOT_DEFAULT="$(pwd)/data/multi_object_open/lerobot"
OUTPUT_DEFAULT="multi_object_open_all_merged"

DATASET_ROOT="${1:-$DATASET_ROOT_DEFAULT}"
OUTPUT_NAME="${2:-$OUTPUT_DEFAULT}"
DRY_RUN=0
FORCE=0
# Parse optional flags from remaining args (supports --dry-run/-n and --force/-f)
for arg in "${@:3}"; do
    case "$arg" in
        --dry-run|-n) DRY_RUN=1 ;;
        --force|-f) FORCE=1 ;;
    esac
done

# Collect immediate subdirectories, skipping the intended output folder and any directories that include 'merge' in their name
repos=()
for d in "$DATASET_ROOT"/*; do
    if [[ -d "$d" ]]; then
        base=$(basename "$d")
        # Skip if this is exactly the output folder itself or contains 'merge' (to avoid merging previously merged outputs)
        if [[ "$base" == "$OUTPUT_NAME" ]] || [[ "$base" == *"merge"* ]] || [[ "$base" == *"merged"* ]]; then
            echo "Skipping $d (output folder or contains 'merge')"
            continue
        fi
        repos+=("$d")
    fi
done

if [ ${#repos[@]} -eq 0 ]; then
    echo "No subdirectories found in '$DATASET_ROOT' to merge after filtering out '$OUTPUT_NAME' and any 'merge' folders"
    exit 1
fi

# Build repo_ids string as a single-line JSON array (double-quoted)
repo_ids_str="["
for ((i=0;i<${#repos[@]};i++)); do
    esc=$(printf '%s' "${repos[$i]}" | sed 's/"/\\"/g')
    if [ $i -gt 0 ]; then
        repo_ids_str+=","
    fi
    repo_ids_str+="\"${esc}\""
done
repo_ids_str+="]"

# Print summary
echo "Found ${#repos[@]} repositories to merge into '$OUTPUT_NAME':"
for r in "${repos[@]}"; do
    echo "  - $r"
done

echo "repo_ids_str: $repo_ids_str"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "DRY RUN: would run merge into: $DATASET_ROOT/$OUTPUT_NAME"
    echo "DRY RUN: python -m lerobot.scripts.lerobot_edit_dataset --repo_id \"$DATASET_ROOT/$OUTPUT_NAME\" --operation.type merge --operation.repo_ids \"$repo_ids_str\""
    exit 0
fi

# Check if output directory exists
if [ -d "$DATASET_ROOT/$OUTPUT_NAME" ]; then
    if [ "$FORCE" -eq 1 ]; then
        echo "Removing existing output directory '$DATASET_ROOT/$OUTPUT_NAME' (force enabled)"
        rm -rf "$DATASET_ROOT/$OUTPUT_NAME"
    else
        echo "Output directory '$DATASET_ROOT/$OUTPUT_NAME' already exists. Use --force to overwrite or choose a different output name."
        exit 1
    fi
fi

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
