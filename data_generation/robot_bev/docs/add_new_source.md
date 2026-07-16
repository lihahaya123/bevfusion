# Adding a Robot BEV source adapter

A new source owns acquisition and source semantics. It does not own the
canonical storage contract, validation policy, geometry formulas, or
BEVFusion-specific conversion.

## Boundary

Keep these layers separate:

| Layer | May contain | Must not contain |
| --- | --- | --- |
| `schema.py` | Stable constants and pure mask/path rules | Habitat, source, writer, or training imports |
| `writer.py` | Atomic canonical files, manifests, indexes, summaries | Source-specific parsing or simulator state |
| `validator.py` | Dependency-light strict validation and quality warnings | Source SDKs or repair logic |
| `geometry_checks.py` | Canonical projection/BEV/sweep diagnostics | Source SDKs |
| `sources/habitat_common.py` | Reusable Habitat sensor, pose, navigation, and depth logic | Replica-only asset or category assumptions |
| `sources/<source>.py` | Asset preflight, source mapping, rendering/conversion orchestration | Private schema variants or BEVFusion fields |
| `cli/` | Argument parsing and orchestration | Duplicate geometry, schema, or writer logic |

A Habitat adapter should reuse the common camera intrinsic, exact sensor
extrinsic, map/base transform, depth-to-point, observation-mask, navigation,
and navmesh helpers. A non-Habitat adapter supplies equivalent canonical values
without importing Habitat.

The adapter constructs one `RobotBEVWriter` for the complete requested scene
set. For every frame it supplies one `FramePayload`, then calls
`write_frame`. It finalizes each successful scene and finalizes the root only
when all requested scenes succeed. Framework-specific `cams`, quaternions,
detection annotations, and sweep records do not belong in this layer.

## Semantic mapping requirements

Define a deterministic source-category policy before rendering:

- map known categories into the fixed order `floor, carpet, obstacle, wall,
  furniture, other`;
- list categories intentionally ignored because they carry no supervision;
- map remaining valid but unknown categories to `other`, rather than silently
  discarding them;
- emit source instance IDs only in supported artifact dtypes;
- set `class_validity` to zero for any canonical class the source cannot
  supervise;
- record mapping groups, ignored categories, and other label-affecting settings
  in the generation fingerprint.

Reject a scene when required semantic metadata is absent or when no usable
instance mapping is loaded. Do not reinterpret unsupported or unobserved cells
as negatives.

## Adapter checklist

1. Preflight every required source asset and name failures with scene context.
2. Define scene-level, disjoint train/validation/test assignments.
3. State source axes, units, timestamp semantics, and exact transforms into
   x-forward/y-left/z-up base coordinates and OpenCV optical camera coordinates.
4. Produce finite `float32` points, intrinsics, and transforms; binary `uint8`
   labels and masks; and only documented optional artifacts.
5. Use root-relative canonical paths through `RobotBEVWriter`; do not write
   private manifests or indexes.
6. Include all training-affecting parameters in the generation fingerprint and
   define deterministic resume behavior if resume is supported.
7. Add unit tests for mapping, coordinate conversion, split leakage, argument
   rejection, orchestration, and dependency boundaries.
8. Generate a small real-source root and run strict complete-root and per-split
   validation.
9. Generate and visually inspect at least one three-image geometry bundle for
   the new adapter. When multiple splits are used, sample each split.
10. Pass single-sample, one-batch forward/backward, and small-overfit gates
    before a production render or training run.

Human geometry review is mandatory for every new adapter even when all numeric
tests pass. It is the release gate for axis handedness, camera projection, BEV
orientation, and historical alignment.
