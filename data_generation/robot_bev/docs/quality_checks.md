# Validation and geometry quality checks

`validate_dataset` is strict and read-only except when a geometry scene is
requested. It prints one JSON object to stdout on success. A validation or
diagnostic failure exits 1 and prints a JSON object with `valid: false`, an
error type, and contextual error text to stderr.

Run validation on the complete root first, then on `train`, `val`, and `test`
separately. Split validation still reconciles the selected split with root and
scene metadata; it is not a replacement for complete-root validation.

## Fatal errors

A fatal error makes the JSON report invalid and blocks conversion or training.
The validator treats these categories as fatal:

| Category | Examples |
| --- | --- |
| Root/schema | Missing root files; wrong schema name/version, class order, BEV contract, or point dimensions |
| Splits | Unknown split keys, duplicate scenes, overlap, invalid scene IDs, or index scenes outside their split |
| Paths/artifacts | Absolute/noncanonical/escaping paths, symlink escape, missing files, malformed PNG/NumPy/point artifacts |
| Records | Duplicate or malformed tokens, noncontiguous frame IDs, bad `prev_token`, nonincreasing timestamps, inconsistent optional paths |
| Numeric | Wrong shape/dtype, nonfinite arrays, nonbinary masks, malformed intrinsics, improper transforms, invalid `pose_valid` |
| Reconciliation | Manifest/index disagreement, manifest artifact metadata mismatch, scene/root summary mismatch, wrong counts or incomplete status |

Fatal messages include dataset, scene, frame, and field context wherever those
values are available. Fix the producer or discard only a separately approved
bad build; the validator does not repair data.

## Warnings

Warnings leave `valid: true` but require review before training:

| Warning | Current threshold |
| --- | --- |
| Empty canonical class | Class total is zero within the validated split |
| Low observed coverage | Mean observed coverage is below 1% |
| Point-count outlier | A frame is below one fifth or above five times the split median; when median is zero, any nonzero frame is flagged |
| Intensity range | Any frame contains intensity outside `[0, 1]` |

Do not suppress warnings solely to obtain a clean log. Explain expected warnings
for a source or investigate mapping, observation masks, and point generation.

## Geometry bundle

For `<scene>/<frame>`, geometry checks write three RGB PNGs below
`<root>/diagnostics/<scene>/`:

| Artifact | What to inspect |
| --- | --- |
| `<frame>_rgb_point_overlay.png` | LiDAR points project onto corresponding visible surfaces with the OpenCV intrinsic/extrinsic convention |
| `<frame>_bev_overlay.png` | x-forward/y-left axes, point cells, camera frustum, observation mask, and class layers agree |
| `<frame>_aligned_sweeps.png` | Static structure from history aligns with the current frame after pose/extrinsic transforms |

File existence, nonzero size, and RGB mode are automated sanity checks. They do
not replace visual approval. Blank-looking or implausibly uniform images,
mirrored axes, systematic projection offsets, and colored sweep ghosts are gate
failures.

Inspect at least one bundle for every new source adapter. For a train/validation/
test dataset, inspect a representative scene from every split. Record the exact
paths, frame IDs, validator JSON, and reviewer decision in the generation
report.

## Release gate

A dataset can proceed to conversion only when:

1. the root summary reports `status: complete` and expected counts;
2. complete-root and every-split validation report `valid: true`;
3. all warnings are resolved or explicitly accepted with evidence;
4. diagnostic files are present and nonblank;
5. a human approves camera projection, x-forward/y-left BEV orientation, and
   historical alignment.

After conversion, independently gate one sample through the data pipeline, one
finite forward/backward batch, and a 16–32-frame overfit before full training.
