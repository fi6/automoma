# Maintainer Tools

The public workflow should start from `scripts/plan.py` or `scripts/run_pipeline.sh`. Everything under `tools/` is first-party but intended for maintainers, diagnostics, release preparation, or paper-specific workflows.

## Directory Map

| Directory | Purpose |
| --- | --- |
| `tools/assets/` | Object and scene asset preparation helpers. |
| `tools/dataset/` | HDF5 layout conversion, trajectory preparation, and dataset visualization. |
| `tools/eval/` | Evaluation helpers called by `scripts/run_pipeline.sh`. |
| `tools/robotwin/` | RoboTwin training and evaluation wrappers. |
| `tools/debug/` | Planning, replay, eval-alignment, plotting, ablation, and validation diagnostics. |
| `tools/dev/` | Local development helpers such as cuRobo build cleanup and smoke pipeline launchers. |
| `tools/ops/` | Long-running job supervision and log guard utilities. |
| `tools/release/` | Dataset release preparation utilities for AutoMoMa-30K and AutoMoMa-500K. |
| `tools/paper/` | Paper-specific reproduction or legacy evaluation scripts. |

## Common Maintainer Commands

```bash
# Clean cuRobo editable-build artifacts before rebuilding.
bash tools/dev/clean_curobo_build.sh

# Visualize raw AutoMoMa HDF5 recordings.
python tools/dataset/viz_hdf5.py data/automoma/<dataset>.hdf5

# Convert split per-episode HDF5 layout to merged layout.
python tools/dataset/convert_hdf5_layout.py <input_dir> <output.hdf5> --direction merge

# Run replay metrics for grasp-filter diagnostics.
python tools/debug/run_grasp_filter_metrics.py --max-objects 5 --scenes-per-object 1 --keep-going
```

## Public Hygiene Rules

- Do not add machine-specific absolute paths to scripts, tools, or docs.
- Use `REPO_ROOT`-relative paths for repository files.
- Use `PYTHON_BIN=${PYTHON_BIN:-python}` in shell wrappers instead of a fixed conda path.
- Keep one-off local recovery scripts out of the public repository.
- Do not put workflow documentation in `scripts/` or `tools/`; add it under `docs/`.
