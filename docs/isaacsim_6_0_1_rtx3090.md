# Isaac Sim 6.0.1 on RTX 3090

This note records the September 8, 2026 bring-up on an eight-GPU RTX 3090
host. It distinguishes a basic RTX renderer check from the full AutoMoMa
recording gate. Passing imports or the synthetic renderer check alone does not
authorize a production migration.

## Pinned source and image

- AutoMoMa branch: `codex/isaacsim-6.0.1`
- AutoMoMa commit tested: `a70725e3a48bc0e19a4a30f0afd8f184be0c2af6`
- IsaacLab-Arena commit: `f4a27c9867236bfed5199d4b911b730bfd7c5604`
- Image tag: `automoma:isaacsim6.0.1`
- Image ID tested: `sha256:39bdf095fa9d64a5093131a7810b0d26a89ca358d6f1445d2f375e7994272757`
- Host driver: `570.207`
- GPU: NVIDIA GeForce RTX 3090, 24 GB

The NVIDIA compatibility checker passed this host. The image import check
reported:

```text
2.11.0+cu128 (2, 28, 9) 2.3.1 NVIDIA GeForce RTX 3090
imports ok
```

## Build and import check

Use a separate checkout so an existing Isaac Sim 5.1 environment remains
available for rollback:

```bash
git clone --branch codex/isaacsim-6.0.1 --recurse-submodules \
  git@github.com:fi6/automoma.git automoma-isaacsim601
cd automoma-isaacsim601
git submodule update --init --recursive
bash docker/build_docker.sh --tag automoma:isaacsim6.0.1
```

If the RoboTwin HTTPS submodule URL requires interactive credentials, change
only this checkout's submodule URL to SSH, then synchronize and update:

```bash
git config submodule.third_party/RoboTwin.url git@github.com:chang-xinhai/RoboTwin.git
git submodule sync --recursive
git submodule update --init --recursive
```

Run the import check on the target GPU:

```bash
docker run --rm --gpus device=0 \
  --entrypoint /isaac-sim/python.sh \
  automoma:isaacsim6.0.1 \
  -c 'import torch,numpy; print(torch.__version__, torch.cuda.nccl.version(), numpy.__version__, torch.cuda.get_device_name(0)); import automoma,isaaclab,isaaclab_arena,curobo; print("imports ok")'
```

## Synthetic RTX render gate

Run this after the import check. `NVIDIA_DRIVER_CAPABILITIES=all` is required
for the graphics path tested here.

```bash
mkdir -p /tmp/automoma601-render-smoke
docker run --rm --gpus device=0 --shm-size=2g \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$PWD/docker/smoke_isaacsim_render.py:/tmp/smoke_isaacsim_render.py:ro" \
  -v /tmp/automoma601-render-smoke:/output \
  --entrypoint /isaac-sim/python.sh \
  automoma:isaacsim6.0.1 \
  /tmp/smoke_isaacsim_render.py --output-dir /output
```

The RTX 3090 host produced a 320x240 RGBA PNG with RGB standard deviation
approximately `[88.11, 86.62, 87.97]` and exited successfully.

## Full recording gate: currently not passed

The full gate uses a real robot, object, materialized scene, finalized
trajectory, physics drive, the contract-declared step count, and all production RGB/depth cameras. On
the tested commit it did not reach scene construction or write HDF5.

Observed blockers, in order:

1. Downstream RLMoMa still imports
   `isaaclab_arena.examples.example_environments.cli`; the pinned Arena commit
   exposes the CLI as `isaaclab_arena_environments.cli`.
2. The image installs the IsaacLab core editable package with `--no-deps` and
   does not contain all packages required by the Arena recording import graph.
3. The missing runtime modules observed while advancing the smoke were
   `prettytable`, `zmq` (PyZMQ), `isaaclab_teleop`, `hydra`, and `pinocchio`.
   `isaaclab_visualizers` was also not installed and emitted an extension
   warning.
4. `python.sh`/Kit returned status zero for several runs that printed an
   uncaught Python traceback. Production validation must therefore require the
   expected summary and HDF5 contract and scan fatal log signatures; container
   status alone is insufficient.

`python -m pip check` also reports a wider set of absent or version-conflicting
IsaacLab and LeRobot dependencies. Do not fix the list by installing all
latest transitive dependencies without pins: that can replace the Torch,
NumPy, packaging, or Hugging Face versions intentionally selected for Isaac
Sim 6.0.1.

## Required migration sequence

Before changing the production image digest:

1. Add a pinned IsaacLab/Arena runtime dependency set to the Docker build and
   extend the build-time check beyond top-level imports.
2. Update downstream Arena CLI imports for the new module layout, preferably
   with an explicit compatibility adapter during rollback testing.
3. Run one real contract-length episode and validate every state/action/RGB/depth
   dataset plus representative images.
4. Run one eight-environment round.
5. Run the exact production logical job: five ordered eight-environment rounds
   and one 40-episode HDF5.
6. Repeat the complete job on every target host, using staggered startup, and
   visually sample all three cameras.
7. Pin the passing image by immutable digest in the production configuration;
   retain the 5.1 image until 6.0.1 recording and recovery are proven.

The private Microwave paths, frozen input identities, and reusable full-smoke
harness are kept in the RLMoMa repository's dated migration handoff rather
than this public document.
