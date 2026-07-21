# Replica v1 generation runbook

This adapter supports original Replica v1 PTex assets through Habitat-Sim
0.2.2. ReplicaCAD and other scene layouts are intentionally unsupported.

## Prerequisites

Activate the `habitat022` environment and provide a discoverable
`replica.scene_dataset_config.json`. Each requested scene must have its render
mesh, PTex parameters and color atlases, Habitat sorted faces, semantic mesh,
`info_semantic.json`, navmesh, and stage configuration. The preflight rejects a
stage configuration that does not point to the expected Replica assets.

Set portable path variables for the machine running the render:

```text
REPLICA_CONFIG=/path/to/replica/replica.scene_dataset_config.json
OUTPUT_ROOT=/path/to/output/replica_robot_bev_v4
```

The tracked scene list has all 18 official scenes. The tracked split is 14
train, 2 validation, and 2 test scenes. In particular, `office_0` is train,
`office_1` is validation, and `office_4` is test.

## One-scene 10-frame quick check

Use this to verify the environment and renderer before a multi-scene job. The
full split file assigns `office_1` to validation; the generated root is filtered
to the requested scene.

```bash
conda activate habitat022
REPLICA_CONFIG=/path/to/replica/replica.scene_dataset_config.json
QUICK_ROOT=/path/to/output/replica_robot_bev_v4_quick
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v4_quick \
  --scene office_1 \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$QUICK_ROOT" \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh
```

Validate it and create one geometry bundle:

```bash
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$QUICK_ROOT" \
  --split val \
  --geometry-scene office_1 \
  --geometry-frame 0
```

## Exact 9-scene × 10-frame training-link smoke

This is the canonical 90-frame smoke dataset. The exact requested subset is
seven train scenes (70 frames), one validation scene (10 frames), and one test
scene (10 frames).

```bash
conda activate habitat022
REPLICA_CONFIG=/path/to/replica/replica.scene_dataset_config.json
OUTPUT_ROOT=/path/to/output/replica_robot_bev_v4
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v4 \
  --scenes hotel_0 office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$OUTPUT_ROOT" \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh
```

Validate the complete root, then each split. These split commands also create
the required train/validation/test geometry samples:

```bash
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT"
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split train \
  --geometry-scene office_0 \
  --geometry-frame 0
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split val \
  --geometry-scene office_1 \
  --geometry-frame 0
python -m data_generation.robot_bev.cli.validate_dataset \
  --root "$OUTPUT_ROOT" \
  --split test \
  --geometry-scene office_4 \
  --geometry-frame 0
```

Success means `multi_scene_summary.json` has `status: complete`, counts are
70/10/10, all four validation reports have `valid: true`, and a human approves
all three geometry bundles. Nonblank PNGs alone do not establish correct
x-forward/y-left orientation or camera projection.

## Remote 18-scene × 600-frame production render

Run production only after the 90-frame validation, geometry, conversion,
single-batch, and overfit gates pass. Choose a new empty build root on the
remote machine; do not overwrite the approved smoke root in place.

```bash
conda activate habitat022
REPLICA_CONFIG=/path/to/replica/replica.scene_dataset_config.json
PRODUCTION_ROOT=/path/to/empty/output/replica_robot_bev_v4
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset "$REPLICA_CONFIG" \
  --dataset-id replica_robot_bev_v4 \
  --scenes-file data_generation/robot_bev/configs/replica_scenes.txt \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir "$PRODUCTION_ROOT" \
  --num-frames 600 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh
```

The expected production counts are 8,400 train, 1,200 validation, and 1,200
test frames. Repeat complete-root, per-split, and representative geometry
validation against `PRODUCTION_ROOT` before promotion to training.

## Resume and failure policy

Use `--resume` only after the output root contains complete, atomically
committed manifest frames from the same command contract. Keep the same dataset
ID, requested scene order, split contents, frame count, sensor settings,
trajectory navmesh settings, BEV label source, BEV label bounds, semantic
mapping, and Habitat-Sim version. The writer rejects
metadata, split, or generation-fingerprint changes.

Resume deterministically replays committed poses and collision recovery from
the original seed, verifies every existing frame ID, timestamp, pose, and
artifact, and continues at the first uncommitted frame. It does not repair a
manually edited manifest or silently accept missing artifacts.

If generation is interrupted after complete frames, report the partial root and
rerun the same generation command with the additional `--resume` argument. If a
scene fails, inspect `multi_scene_summary.json` and the original error before
taking action. Do not delete a partial root merely to make a rerun convenient,
and never combine outputs from different fingerprints.

The legacy generation-time visualization and PLY switches are rejected by the
canonical adapter. Generate camera, BEV, and aligned-sweep diagnostics afterward
with the validator as shown above.

## Replica semantic mapping

The source adapter normalizes Replica category names, maps known categories to
the six fixed schema classes, omits explicitly ignored non-supervision
categories, and maps remaining nonignored categories to `clutter`. The complete
mapping groups and ignored set are included in the generation fingerprint.
Habitat-Sim 0.2.2 must load the semantic PLY with y-up/-z-forward orientation;
the adapter overrides the z-up PTex stage defaults for the semantic asset so
that RGB, depth, and instance IDs remain registered. Rendering is fatal if the
semantic scene yields zero instance mappings, the front semantic observation
is empty, semantic coverage over valid depth is below the configured threshold,
or an instance ID cannot fit `uint16`.

BEV semantic labels are generated from semantic IDs carried by projected depth
points inside the canonical x/y/z label bounds for every class, including
`floor` and `carpet`. The navmesh remains part of trajectory sampling/replay,
but it is not used to fill `floor`, mark obstacles, or inject agent-radius
traversability into semantic labels. Ceiling and ceiling-mounted fixtures are
ignored rather than mapped to `clutter`.
