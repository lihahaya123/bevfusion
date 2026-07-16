# Robot BEV dataset schema v3

Schema v3 is the stable, framework-independent contract emitted by
`RobotBEVWriter`. A coordinate convention, class order, tensor shape, required
field, path interpretation, or unit change requires a new schema version.

## Root contract

Every dataset root contains:

```text
<root>/
  dataset_metadata.json
  splits.json
  multi_scene_summary.json
  robot_infos_train.pkl
  robot_infos_val.pkl
  robot_infos_test.pkl
  <scene_id>/
    images/
    points/
    bev_masks/
    bev_observed_masks/
    calib/
    poses/
    manifest.jsonl
    metadata.json
    summary.json
    scene_infos.pkl
    robot_infos_<split>.pkl
    depths/                    # optional
    semantics/                 # optional
    bev_supervision_masks/     # optional
```

All stored artifact paths are normalized POSIX paths relative to the dataset
root. Absolute paths, `..`, noncanonical spellings, path escapes, and symlink
escapes are invalid. Scene IDs are nonempty normalized relative paths and occur
in exactly one of `train`, `val`, or `test`.

`dataset_metadata.json` fixes these values:

| Field | Value |
| --- | --- |
| `schema_name` | `robot_bev_dataset` |
| `schema_version` | `3` |
| `map_classes` | `floor, carpet, obstacle, wall, furniture, other` |
| BEV x bound | `[0.0, 3.0, 0.02]` metres |
| BEV y bound | `[-1.5, 1.5, 0.02]` metres |
| BEV label shape | `(6, 150, 150)`, binary `uint8` |
| observed-mask shape | `(150, 150)`, binary `uint8` |
| point dtype | `float32` |
| point dimensions | `x, y, z, intensity, time` |

Each root also records a generation fingerprint. It covers the schema and
training-affecting generation parameters, including requested scenes, split
contents, sensor settings, and semantic mapping. Resume must match it exactly.

## Frame record

Each root and scene pickle has `metadata` and an ordered `infos` list. Each
manifest JSON object and corresponding info record contains:

| Field | Contract |
| --- | --- |
| `dataset_id`, `scene_id` | Nonempty identifiers; no `:` |
| `frame_id` | Zero-based contiguous integer within a scene |
| `token` | `<dataset_id>:<scene_id>:<frame_id six digits>`; globally unique |
| `prev_token` | Empty on frame zero; otherwise the previous frame in the same scene |
| `timestamp` | Integer microseconds; strictly increasing within a scene |
| `image_path` | Required RGB PNG path |
| `lidar_path` | Required binary `float32` point path |
| `bev_mask_path` | Required six-channel binary `uint8` NumPy path |
| `bev_observed_mask_path` | Required binary `uint8` NumPy path |
| `bev_supervision_mask_path` | Optional per-class binary `uint8` NumPy path |
| `depth_path` | Optional `uint16` millimetre depth PNG path |
| `semantic_path` | Optional `uint16` instance-ID PNG path |
| `class_validity` | Binary `uint8` vector with shape `(6,)` |
| `cam_intrinsic` | Finite `float32` 3×3 pinhole intrinsic matrix |
| `camera2base` | Finite `float32` 4×4 optical-camera-to-base transform |
| `lidar2base` | Finite `float32` 4×4 LiDAR-to-base transform |
| `T_map_base` | Finite `float32` 4×4 base-to-map transform |
| `pose_valid` | JSON/Python boolean; formal generated data is `true` |

Transform rotations are orthonormal, have determinant +1, and use homogeneous
bottom row `[0, 0, 0, 1]`. Pickle arrays retain the listed NumPy dtypes; JSON
manifests encode the same values as ordinary lists.

## Coordinates and transforms

All transforms multiply column vectors. `A2B` means `T_B_from_A`:

```text
p_base = camera2base @ p_camera_optical
p_base = lidar2base  @ p_lidar
p_map  = T_map_base @ p_base
```

Base and LiDAR coordinates are right-handed: x forward, y left, z up, in
metres. Camera coordinates use the OpenCV optical convention: x right, y down,
z forward. Habitat/OpenGL axes are converted before `camera2base` is stored.

The BEV row follows forward x and the column follows left y:

```text
row = floor((x_forward - 0.0) / 0.02)
col = floor((y_left + 1.5) / 0.02)
```

The stored array is not an image-coordinate promise; use these formulas for
training and use the diagnostic overlay to verify display orientation.

Point `intensity` is expected in `[0, 1]`. Point `time` is seconds relative to
the current frame: zero for current points and nonpositive for aligned history.
Canonical source data stores no BEVFusion-specific sweeps; a converter derives
history transforms from poses and extrinsics.

## Labels and supervision

BEV labels are multi-hot: more than one class channel may be one at a cell. The
class order never changes within schema v3:

1. `floor`
2. `carpet`
3. `obstacle`
4. `wall`
5. `furniture`
6. `other`

`bev_observed_mask` identifies cells with sensor evidence. `class_validity`
identifies classes that the source can supervise. An optional per-class mask
can further restrict regional supervision. The effective training mask is:

```text
observed_mask[None, :, :]
  * class_validity[:, None, None]
  * optional_per_class_supervision_mask
```

Never convert unobserved or unsupported cells into negative labels.
