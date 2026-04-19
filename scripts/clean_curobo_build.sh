#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
curobo_dir="$repo_root/third_party/curobo"

if [[ ! -d "$curobo_dir" ]]; then
  echo "curobo not found at: $curobo_dir" >&2
  exit 1
fi

# Remove common build artifacts from curobo editable builds.
find "$curobo_dir" -maxdepth 2 -type d \( -name "build" -o -name "dist" -o -name "*.egg-info" \) -prune -exec rm -rf {} +

# Remove compiled extensions produced under curobo sources.
find "$curobo_dir/src/curobo" -type f \( -name "*.so" -o -name "*.pyd" -o -name "*.dll" \) -delete

echo "Cleaned curobo build artifacts in $curobo_dir"