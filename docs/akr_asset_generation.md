# AKR Asset Generation

`tools/assets/build_akr_assets.py` rebuilds the Summit Franka + articulated
object assets used by cuRobo planning. It does not generate grasps. Its minimum
object input is:

- a prepared and scaled object URDF;
- `init_state.npz`, including `data.object.scaling` and the initial joint state;
- a directory of 7-D grasp NPY files in `[x, y, z, qw, qx, qy, qz]` order.

The tool runs on CPU. cuRobo, CUDA, Isaac Sim, and matplotlib are not imported
by construction or sphere generation.

## Install

Use Python 3.11 and install AutoMoMa from the repository root:

```bash
conda create -n automoma-assets python=3.11 -y
conda activate automoma-assets
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The dependency bounds intentionally retain NumPy 1.26, SciPy below 1.16, and
`yourdfpy` 0.0.57–0.0.60. Those versions reproduce the checked-in XML
floating-point serialization. Newer numeric stacks remain semantically correct
but can differ at approximately `1e-17`, which changes SHA256.

Confirm the CPU path before generating:

```bash
python -c "import numpy, scipy, trimesh, yaml, yourdfpy; print(numpy.__version__, scipy.__version__, yourdfpy.__version__)"
pytest -q tests/test_akr_assets.py -m "not gpu"
```

## Reproduce the checked-in 7221 assets

Run this from the AutoMoMa repository root. The output is a staging directory;
the command refuses to write to a non-empty directory unless `--force` is
explicitly supplied.

```bash
python tools/assets/build_akr_assets.py \
  --object-urdf assets/object/Microwave/7221/7221_0_scaling.urdf \
  --object-init-state assets/object/Microwave/7221/0/init_state.npz \
  --grasp-dir assets/object/Microwave/7221/grasp \
  --grasp-ids 0-20 \
  --object-base link_1 \
  --handle-link link_0 \
  --object-joint joint_0 \
  --robot-urdf assets/robot/summit_franka/summit_franka.urdf \
  --robot-yaml assets/robot/summit_franka/summit_franka.yml \
  --robot-fixed-yaml assets/robot/summit_franka/summit_franka_fixed_base.yml \
  --sphere-yaml assets/object/Microwave/7221/summit_franka_7221_0_grasp_0000.yml \
  --asset-prefix assets/object/Microwave/7221 \
  --output-dir /tmp/automoma-7221-akr
```

When `--sphere-yaml` points at a checked-in combined config, the tool extracts
the object sphere block and uses the corresponding per-grasp frozen profile.
This preserves the historical self-collision list ordering and makes all 63
formal outputs byte-identical. `collision_spheres.yml` is an additional
standalone convenience output and is not one of the 63 golden files.

Verify the formal outputs:

```bash
for generated in /tmp/automoma-7221-akr/summit_franka*; do
  golden="assets/object/Microwave/7221/$(basename "$generated")"
  sha256sum "$generated" "$golden"
done
```

Only after reviewing the staging directory should existing assets be replaced:

```bash
python tools/assets/build_akr_assets.py \
  ...same arguments... \
  --output-dir assets/object/Microwave/7221 \
  --force
```

The unreferenced `summit_franka_7221_0_grasp_0000_05.urdf` debug asset is not
generated or compared.

## Regenerate collision spheres

Replace `--sphere-yaml ...` with `--generate-spheres`:

```bash
python tools/assets/build_akr_assets.py \
  --object-urdf assets/object/Microwave/7221/7221_0_scaling.urdf \
  --object-init-state assets/object/Microwave/7221/0/init_state.npz \
  --grasp-dir assets/object/Microwave/7221/grasp \
  --grasp-ids 0-20 \
  --robot-urdf assets/robot/summit_franka/summit_franka.urdf \
  --robot-yaml assets/robot/summit_franka/summit_franka.yml \
  --robot-fixed-yaml assets/robot/summit_franka/summit_franka_fixed_base.yml \
  --generate-spheres \
  --asset-prefix assets/object/Microwave/7221 \
  --output-dir /tmp/automoma-7221-akr-spheres
```

The fixed 7221 parameters are:

| Link | Meaning | Radius | Spacing factor | Expected count |
| --- | --- | ---: | ---: | ---: |
| `link_0` | handle/door | `0.019` | `3.0` | about `90` |
| `link_1` | body | `0.049` | `2.0` | about `80` |

The accepted golden counts are 89 and 87; regenerated counts must stay within
10%. A missing mesh, geometry-free requested link, unsupported primitive, or
empty output is a hard error. Meshes are found from the URDF location and its
ancestors, so generation does not depend on the current working directory.

## Use a new object with an existing robot

1. Prepare and scale the object URDF. Preserve its articulated joint and give
   the body and grasped handle unambiguous link names.
2. Export `init_state.npz` with `data.object.scaling` and `data.object.qpos`.
3. Export grasp NPY files in WXYZ quaternion order. Translation is stored in
   unscaled object coordinates; the builder applies `scaling`.
4. Choose the object body (`--object-base`), grasped link (`--handle-link`),
   and movable joint (`--object-joint`).
5. First run with `--generate-spheres` into a fresh staging directory.
6. Load the generated URDF and YAML in cuRobo, inspect collision coverage, and
   tune sphere parameters in `automoma/assets/akr.py` if the new object's
   dimensions differ materially from 7221.
7. Copy the reviewed outputs into the object's asset directory, retaining the
   same `--asset-prefix`.

The attachment is
`T_ee_handle = inverse(T_object_grasp) @ T_object_handle`. The builder reverses
the object chain so the handle is the attached root, connects its former world
joint to `ee_link`, appends the movable object joint to all cspace arrays, and
sets the cuRobo end-effector link to the object body.

## Shared asset layout

For a read-mostly shared checkout, copy the code into a personal directory and
link only large assets:

```bash
cp -a /data/shared/automoma /data/$USER/automoma
cd /data/$USER/automoma
rm -rf assets
ln -s /data/shared/automoma/assets assets
```

Generate into a personal staging directory. Do not pass `--force` against the
shared asset tree unless you are the designated maintainer and have reviewed
the complete staged diff.

## Optional GPU validation

After CPU tests pass, use the normal AutoMoMa planning environment to load
mobile and fixed configs for grasp `0000` and `0020`, run FK, and execute a
collision query. See the repository planning installation and `scripts/plan.py`.
Isaac Sim/cuRobo installation is not needed for asset construction itself.
