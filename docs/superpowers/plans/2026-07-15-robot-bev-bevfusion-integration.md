# Robot BEV BEVFusion Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert canonical Robot BEV datasets into portable BEVFusion indexes and train/evaluate camera-plus-LiDAR segmentation with masked supervision and selective nuScenes pretraining.

**Architecture:** This is plan 2 of 2 and starts only after the data-toolkit completion gate. One deterministic converter adapts canonical records to the existing NuScenesDataset consumer; dataset/pipeline changes resolve root-relative paths and combine supervision masks. Model, evaluation, checkpoint loading, and a dedicated configuration complete an end-to-end smoke-to-remote-training workflow.

**Tech Stack:** Python 3.8, PyTorch 1.9–1.10.2, MMCV 1.4.0, MMDetection 2.20.0, torchpack, NumPy, pytest, compiled BEVFusion CUDA extensions.

## Global Constraints

- Plan 1 must have produced a canonical schema-v3 dataset and passed structural, numeric, and human geometry validation.
- Classes are exactly `floor`, `carpet`, `obstacle`, `wall`, `furniture`, `other` in that order.
- Label output is exactly `uint8 [6,150,150]` for `[0.0,3.0] × [-1.5,1.5]` at 0.02 m.
- Dataset paths resolve only through the explicit `dataset_root`; default nuScenes path behavior remains unchanged.
- Effective supervision is observed-mask broadcast × `class_validity` × optional per-class mask.
- Invalid cells contribute neither focal loss nor TP/FP/FN.
- `unknown` is derived for visualization and is never a learned class.
- Robot segmentation uses the base dataset directly, not `CBGSDataset`.
- Initial training uses current points plus at most five history sweeps.
- BEV-related 3D rotation, scale, translation, and flip remain disabled.
- nuScenes geometry buffers and the final six-class output convolution are never loaded from the checkpoint.
- Checkpoint parameters load only when both normalized name and shape match; unexpected missing core modules fail loudly.
- Fixed `mIoU@0.50` selects checkpoints; threshold-max IoU is diagnostic only; test is evaluated once after selection.
- Execute unit/integration/model tests inside the Python 3.8 BEVFusion environment with compiled ops; the current host Python 3.14 environment is not a valid execution environment.

## File Map

- `tools/data_converter/robot_bev_converter.py`: canonical-to-BEVFusion conversion and history sweeps.
- `mmdet3d/datasets/nuscenes_dataset.py`: explicit root-relative path resolution and masked map evaluation.
- `mmdet3d/datasets/pipelines/loading.py`: Robot BEV label/observed/class/per-class mask loader.
- `mmdet3d/datasets/pipelines/formating.py`: stack labels and supervision masks.
- `mmdet3d/models/heads/segm/vanilla.py`: masked per-class focal loss.
- `mmdet3d/models/fusion_models/bevfusion.py`: optional depths, supervision-mask forwarding, evaluation payload.
- `mmdet3d/utils/checkpoint.py`: name-and-shape selective checkpoint loader/report.
- `mmdet3d/apis/train.py`: resume/selective-load/legacy-load precedence.
- `configs/robot_bev/default.yaml`: canonical dataset and pipeline defaults.
- `configs/robot_bev/seg/camera_lidar_lss.yaml`: robot camera+LiDAR segmentation model and fine-tuning settings.
- `docs/robot_bev_training.md`: local smoke, remote full training, evaluation, and failure recovery commands.
- `tests/test_robot_bev_converter.py`, `tests/test_robot_bev_pipeline.py`, `tests/test_bev_valid_mask.py`, `tests/test_robot_bev_checkpoint.py`, `tests/test_robot_bev_config.py`: TDD coverage.

---

### Task 1: Deterministic canonical-to-BEVFusion converter

**Files:**
- Create: `tools/data_converter/robot_bev_converter.py`
- Test: `tests/test_robot_bev_converter.py`

**Interfaces:**
- Consumes: `validate_dataset()`, canonical `robot_infos_<split>.pkl`, `dataset_root`, and `max_sweeps`.
- Produces: `convert_split(root, split, max_sweeps=5, camera_name="CAM_FRONT") -> Path` and portable `bevfusion_infos_<split>.pkl`.

- [ ] **Step 1: Write converter tests with two frames and one history transform**

```python
# tests/test_robot_bev_converter.py
import pickle

import numpy as np

from tools.data_converter.robot_bev_converter import convert_split


def test_converter_preserves_relative_paths_and_builds_sweeps(canonical_root):
    output = convert_split(canonical_root, "train", max_sweeps=5)
    with output.open("rb") as handle:
        payload = pickle.load(handle)
    first, second = payload["infos"][:2]
    assert first["lidar_path"] == "scene_a/points/000000.bin"
    assert first["cams"]["CAM_FRONT"]["data_path"] == (
        "scene_a/images/000000.png"
    )
    assert first["sweeps"] == []
    assert len(second["sweeps"]) == 1
    np.testing.assert_allclose(
        second["sweeps"][0]["sensor2lidar_translation"],
        np.array([-0.1, 0.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert second["bev_observed_mask_path"].endswith(
        "bev_observed_masks/000001.npy"
    )
    assert payload["metadata"]["version"] == "robot-bev-v3"


def test_converter_is_byte_deterministic(canonical_root):
    first = convert_split(canonical_root, "train", max_sweeps=5).read_bytes()
    second = convert_split(canonical_root, "train", max_sweeps=5).read_bytes()
    assert first == second
```

Provide `canonical_root` in `tests/conftest.py` by using Plan-1 `RobotBEVWriter` to create two `scene_a` frames; set the second `T_map_base[0,3]=0.1`.

- [ ] **Step 2: Run the tests and verify converter import failure**

Run: `pytest -q tests/test_robot_bev_converter.py`

Expected: FAIL with `No module named 'tools.data_converter.robot_bev_converter'`.

- [ ] **Step 3: Implement converter transforms and public API**

Use this exact conversion formula and output contract:

```python
def _history_to_current(cur, hist):
    cur_map_from_lidar = cur["T_map_base"] @ cur["lidar2base"]
    hist_map_from_lidar = hist["T_map_base"] @ hist["lidar2base"]
    return np.linalg.inv(cur_map_from_lidar) @ hist_map_from_lidar


def _make_sweeps(current, history, max_sweeps):
    sweeps = []
    for hist in list(history)[-max_sweeps:][::-1]:
        transform = _history_to_current(current, hist)
        sweeps.append({
            "data_path": normalize_relative_path(hist["lidar_path"]),
            "timestamp": int(hist["timestamp"]),
            "sensor2lidar_rotation": transform[:3, :3].astype(np.float32),
            "sensor2lidar_translation": transform[:3, 3].astype(np.float32),
        })
    return sweeps


def convert_split(root, split, max_sweeps=5, camera_name="CAM_FRONT"):
    root = Path(root).expanduser().resolve()
    validate_dataset(root, split)
    source = _load_pickle(root / f"robot_infos_{split}.pkl")
    history_by_scene = {}
    converted = []
    for raw in source["infos"]:
        scene_history = history_by_scene.setdefault(raw["scene_id"], [])
        converted.append(
            _convert_frame(raw, scene_history, max_sweeps, camera_name)
        )
        scene_history.append(raw)
    metadata = dict(source["metadata"])
    metadata.update({
        "version": "robot-bev-v3",
        "converter": "robot_bev_converter_v1",
        "source_schema_name": "robot_bev_dataset",
        "source_schema_version": 3,
    })
    output = root / f"bevfusion_infos_{split}.pkl"
    _atomic_pickle(output, {"infos": converted, "metadata": metadata})
    return output
```

`_convert_frame()` must:

- use `camera2base` directly without OpenGL/OpenCV flipping;
- compute `camera2lidar = inverse(lidar2base) @ camera2base`;
- write camera sensor2ego from `camera2base`, lidar2ego from `lidar2base`, and ego2global from `T_map_base`;
- convert rotations to `[w,x,y,z]` quaternions with a local deterministic helper;
- keep every data/mask path root-relative;
- copy `class_validity` and optional supervision-mask path;
- generate empty float32/int64/bool 3D annotations;
- build sweeps only from earlier frames in the same scene.

- [ ] **Step 4: Add CLI and all-split conversion**

Add parser options `--root`, `--split {train,val,test,all}`, `--max-sweeps`, and `--camera-name`. `all` converts the three split indexes in train/val/test order and exits nonzero if validation fails.

Run: `python tools/data_converter/robot_bev_converter.py --help`

Expected: exit 0 and list all four arguments.

- [ ] **Step 5: Run converter tests and commit**

Run: `pytest -q tests/test_robot_bev_converter.py`

Expected: all tests PASS.

```bash
git add tools/data_converter/robot_bev_converter.py tests/conftest.py \
  tests/test_robot_bev_converter.py
git commit -m "feat: convert canonical robot BEV datasets"
```

---

### Task 2: Root-relative dataset paths and Robot BEV mask pipeline

**Files:**
- Modify: `mmdet3d/datasets/nuscenes_dataset.py:125-320`
- Modify: `mmdet3d/datasets/pipelines/loading.py:317-335`
- Modify: `mmdet3d/datasets/pipelines/formating.py:95-130`
- Test: `tests/test_robot_bev_pipeline.py`

**Interfaces:**
- Consumes: converter fields `bev_mask_path`, `bev_observed_mask_path`, `class_validity`, and optional `bev_supervision_mask_path`.
- Produces: root-resolved sensor paths, `LoadRobotBEVSegmentation`, `gt_masks_bev`, and stacked `gt_supervision_mask_bev`.

- [ ] **Step 1: Write path and mask-loader tests**

```python
# tests/test_robot_bev_pipeline.py
import numpy as np

from mmdet3d.datasets.nuscenes_dataset import NuScenesDataset
from mmdet3d.datasets.pipelines.loading import LoadRobotBEVSegmentation


def test_dataset_resolves_every_relative_artifact(canonical_converted_root):
    dataset = NuScenesDataset(
        ann_file=str(canonical_converted_root / "bevfusion_infos_train.pkl"),
        pipeline=[],
        dataset_root=str(canonical_converted_root),
        object_classes=[],
        map_classes=["floor", "carpet", "obstacle", "wall", "furniture", "other"],
        modality={"use_camera": True, "use_lidar": True, "use_radar": False},
        test_mode=True,
        filter_empty_gt=False,
        resolve_relative_paths=True,
    )
    data = dataset.get_data_info(0)
    assert data["lidar_path"].startswith(str(canonical_converted_root))
    assert data["image_paths"][0].startswith(str(canonical_converted_root))
    assert data["bev_mask_path"].startswith(str(canonical_converted_root))
    assert data["sweeps"] == []


def test_loader_combines_observed_class_and_optional_masks(tmp_path):
    labels = np.zeros((6, 150, 150), dtype=np.uint8)
    observed = np.ones((150, 150), dtype=np.uint8)
    regional = np.ones((6, 150, 150), dtype=np.uint8)
    regional[4, :, 75:] = 0
    np.save(tmp_path / "labels.npy", labels)
    np.save(tmp_path / "observed.npy", observed)
    np.save(tmp_path / "regional.npy", regional)
    transform = LoadRobotBEVSegmentation(
        classes=("floor", "carpet", "obstacle", "wall", "furniture", "other")
    )
    result = transform({
        "bev_mask_path": str(tmp_path / "labels.npy"),
        "bev_observed_mask_path": str(tmp_path / "observed.npy"),
        "bev_supervision_mask_path": str(tmp_path / "regional.npy"),
        "class_validity": np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8),
    })
    assert result["gt_masks_bev"].shape == (6, 150, 150)
    assert result["gt_supervision_mask_bev"][1].sum() == 0
    assert result["gt_supervision_mask_bev"][4, :, 75:].sum() == 0
```

- [ ] **Step 2: Run tests and verify expected API failures**

Run: `pytest -q tests/test_robot_bev_pipeline.py`

Expected: FAIL because `resolve_relative_paths` and `LoadRobotBEVSegmentation` do not exist.

- [ ] **Step 3: Add explicit root-relative resolution to NuScenesDataset**

Add `resolve_relative_paths=False` to `NuScenesDataset.__init__`, store it, and implement:

```python
def _resolve_data_path(self, value):
    if not self.resolve_relative_paths:
        return value
    root = Path(self.dataset_root).expanduser().resolve()
    raw = Path(str(value).replace("\\", "/"))
    if raw.is_absolute():
        raise ValueError(f"absolute paths are forbidden for robot BEV data: {value}")
    resolved = (root / raw).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes dataset_root: {value}")
    return resolved.as_posix()
```

Use it for current lidar, every sweep `data_path`, every camera `data_path`, BEV label path, observed path, and optional supervision path. Copy each sweep dict before replacing its path so cached `data_infos` are not mutated. Pass `class_validity` through as `np.uint8 [6]`.

- [ ] **Step 4: Replace the old file-only loader with a canonical loader**

Keep `LoadBEVSegmentationFromFile` registered for backward compatibility and add:

```python
@PIPELINES.register_module()
class LoadRobotBEVSegmentation:
    def __init__(self, classes):
        self.classes = tuple(classes)
        if self.classes != (
            "floor", "carpet", "obstacle", "wall", "furniture", "other"
        ):
            raise ValueError(f"unexpected Robot BEV class order: {self.classes}")

    def __call__(self, data):
        labels = np.load(data["bev_mask_path"], allow_pickle=False)
        observed = np.load(data["bev_observed_mask_path"], allow_pickle=False)
        class_validity = np.asarray(data["class_validity"], dtype=np.uint8)
        per_class = None
        if data.get("bev_supervision_mask_path"):
            per_class = np.load(
                data["bev_supervision_mask_path"], allow_pickle=False
            )
        data["gt_masks_bev"] = labels.astype(np.int64, copy=False)
        data["gt_supervision_mask_bev"] = effective_supervision_mask(
            observed, class_validity, per_class
        ).astype(np.int64, copy=False)
        return data
```

- [ ] **Step 5: Stack both BEV tensors in DefaultFormatBundle3D**

Before returning from `DefaultFormatBundle3D.__call__`, add:

```python
for key in ("gt_masks_bev", "gt_supervision_mask_bev"):
    if key in results:
        results[key] = DC(to_tensor(results[key]), stack=True)
```

The config's `Collect3D.keys` must include both names; `Collect3D` will preserve their DataContainers.

- [ ] **Step 6: Run pipeline tests and commit**

Run: `pytest -q tests/test_robot_bev_pipeline.py`

Expected: all tests PASS.

```bash
git add mmdet3d/datasets/nuscenes_dataset.py \
  mmdet3d/datasets/pipelines/loading.py \
  mmdet3d/datasets/pipelines/formating.py tests/test_robot_bev_pipeline.py
git commit -m "feat: load canonical robot BEV supervision"
```

---

### Task 3: Masked focal loss and fusion-model forwarding

**Files:**
- Modify: `mmdet3d/models/heads/segm/vanilla.py:10-120`
- Modify: `mmdet3d/models/fusion_models/bevfusion.py:225-386`
- Test: `tests/test_bev_valid_mask.py`

**Interfaces:**
- Consumes: `gt_masks_bev` and `gt_supervision_mask_bev` tensors `[B,6,150,150]`.
- Produces: per-class masked focal losses, optional `depths`, and inference results containing both GT tensors.

- [ ] **Step 1: Write masked-loss tests**

```python
# tests/test_bev_valid_mask.py
import torch

from mmdet3d.models.heads.segm.vanilla import masked_sigmoid_focal_loss


def test_invalid_target_changes_do_not_change_loss():
    logits = torch.zeros((1, 2, 2), requires_grad=True)
    targets_a = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.float32)
    targets_b = torch.tensor([[[1, 1], [1, 1]]], dtype=torch.float32)
    valid = torch.tensor([[[1, 0], [0, 0]]], dtype=torch.float32)
    loss_a = masked_sigmoid_focal_loss(logits, targets_a, valid)
    loss_b = masked_sigmoid_focal_loss(logits, targets_b, valid)
    torch.testing.assert_close(loss_a, loss_b)


def test_zero_valid_pixels_return_connected_zero():
    logits = torch.randn((1, 2, 2), requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.zeros_like(logits)
    loss = masked_sigmoid_focal_loss(logits, target, valid)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum().item() == 0.0
```

- [ ] **Step 2: Run tests and verify missing function**

Run: `pytest -q tests/test_bev_valid_mask.py`

Expected: FAIL importing `masked_sigmoid_focal_loss`.

- [ ] **Step 3: Implement unreduced focal loss and masked normalization**

```python
def masked_sigmoid_focal_loss(inputs, targets, valid, gamma=2.0):
    inputs = inputs.float()
    targets = targets.float()
    valid = valid.float()
    probability = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = probability * targets + (1 - probability) * (1 - targets)
    raw = ce * ((1 - p_t) ** gamma)
    denominator = valid.sum()
    return (raw * valid).sum() / denominator.clamp_min(1.0)
```

Change `BEVSegmentationHead.forward(x, target=None, supervision_mask=None)` so training requires both tensors and calls the function per class. Return exactly two tensors per class:

```python
losses[f"{name}/focal"] = masked_sigmoid_focal_loss(
    x[:, index], target[:, index], supervision_mask[:, index]
)
losses[f"{name}/valid_pixels"] = supervision_mask[:, index].sum().detach()
```

The existing BEVFusion loop routes tensors with gradients to `loss/map/...` and detached tensors to `stats/map/...`, so no loss-parser change is required.

- [ ] **Step 4: Forward masks through BEVFusion and make depths optional**

Add `gt_supervision_mask_bev=None` to `forward()` and `forward_single()`. Change the required `depths` argument to `depths=None`. Pass both tensors to the map head:

```python
losses = head(x, gt_masks_bev, gt_supervision_mask_bev)
```

During inference attach both GT tensors when present:

```python
if gt_masks_bev is not None:
    result["gt_masks_bev"] = gt_masks_bev[k].cpu()
if gt_supervision_mask_bev is not None:
    result["gt_supervision_mask_bev"] = gt_supervision_mask_bev[k].cpu()
```

- [ ] **Step 5: Run head tests and a model-forward smoke test**

Run: `pytest -q tests/test_bev_valid_mask.py`

Expected: all tests PASS.

Add this concrete head test and assert six finite losses plus successful backward:

```python
head = BEVSegmentationHead(
    in_channels=4,
    grid_transform={
        "input_scope": [(0.0, 3.0, 0.3), (-1.5, 1.5, 0.3)],
        "output_scope": [(0.0, 3.0, 0.02), (-1.5, 1.5, 0.02)],
    },
    classes=["floor", "carpet", "obstacle", "wall", "furniture", "other"],
    loss="focal",
).train()
features = torch.randn((1, 4, 10, 10), requires_grad=True)
target = torch.zeros((1, 6, 150, 150))
valid = torch.ones_like(target)
losses = head(features, target, valid)
assert len(losses) == 12
total = sum(value for name, value in losses.items() if name.endswith("/focal"))
assert torch.isfinite(total)
total.backward()
assert features.grad is not None
```

- [ ] **Step 6: Commit masked training support**

```bash
git add mmdet3d/models/heads/segm/vanilla.py \
  mmdet3d/models/fusion_models/bevfusion.py tests/test_bev_valid_mask.py
git commit -m "feat: mask robot BEV segmentation loss"
```

---

### Task 4: Masked fixed-threshold map evaluation

**Files:**
- Modify: `mmdet3d/datasets/nuscenes_dataset.py:490-535`
- Test: `tests/test_bev_valid_mask.py`

**Interfaces:**
- Consumes: result dictionaries containing predicted probabilities, GT masks, and supervision masks.
- Produces: per-class IoU at thresholds 0.35–0.65, fixed `map/mean/iou@0.50`, and per-class valid-pixel counts.

- [ ] **Step 1: Add an evaluation test proving invalid false positives are ignored**

```python
def test_masked_iou_ignores_invalid_false_positives():
    dataset = object.__new__(NuScenesDataset)
    dataset.map_classes = [
        "floor", "carpet", "obstacle", "wall", "furniture", "other"
    ]
    prediction = torch.zeros((6, 2, 2))
    label = torch.zeros((6, 2, 2))
    valid = torch.zeros((6, 2, 2), dtype=torch.bool)
    prediction[0, 0, 0] = 0.9
    label[0, 0, 0] = 1
    valid[0, 0, 0] = True
    prediction[0, 1, 1] = 0.9
    metrics = dataset.evaluate_map([{
        "masks_bev": prediction,
        "gt_masks_bev": label,
        "gt_supervision_mask_bev": valid,
    }])
    assert metrics["map/floor/iou@0.50"] == 1.0
    assert metrics["map/floor/valid_pixels"] == 1
    assert metrics["map/mean/iou@0.50"] == 1.0
```

- [ ] **Step 2: Run the test and verify current evaluation counts invalid pixels**

Run: `pytest -q tests/test_bev_valid_mask.py::test_masked_iou_ignores_invalid_false_positives`

Expected: FAIL because current evaluation ignores `gt_supervision_mask_bev` and does not emit fixed mean/valid counts.

- [ ] **Step 3: Apply supervision masks to TP/FP/FN**

For each result and threshold compute:

```python
predicted = pred[:, :, None] >= thresholds
truth = label[:, :, None]
valid = supervision[:, :, None]
tp += (predicted & truth & valid).sum(dim=1)
fp += (predicted & ~truth & valid).sum(dim=1)
fn += (~predicted & truth & valid).sum(dim=1)
valid_counts += supervision.sum(dim=1)
```

Report every threshold, explicit fixed `map/mean/iou@0.50`, diagnostic `map/mean/iou@max`, and per-class integer valid counts. For a class with zero valid pixels, emit per-class IoU 0 and a warning rather than NaN; compute mean IoU only across classes with at least one valid pixel.

- [ ] **Step 4: Run evaluation tests and commit**

Run: `pytest -q tests/test_bev_valid_mask.py`

Expected: all tests PASS.

```bash
git add mmdet3d/datasets/nuscenes_dataset.py tests/test_bev_valid_mask.py
git commit -m "feat: mask robot BEV map evaluation"
```

---

### Task 5: Selective checkpoint loading by name and shape

**Files:**
- Create: `mmdet3d/utils/checkpoint.py`
- Modify: `mmdet3d/utils/__init__.py`
- Modify: `mmdet3d/apis/train.py:120-126`
- Test: `tests/test_robot_bev_checkpoint.py`

**Interfaces:**
- Consumes: model, checkpoint path, excluded prefixes, required prefixes.
- Produces: `SelectiveLoadReport`, shape-safe parameter loading, and train precedence `resume_from > selective_load_from > load_from`.

- [ ] **Step 1: Write selective-loader tests with same-shape semantic head**

```python
# tests/test_robot_bev_checkpoint.py
import torch
from torch import nn

from mmdet3d.utils.checkpoint import load_selective_checkpoint


class TinyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({"camera": nn.ModuleDict({"backbone": nn.Linear(2, 2)})})
        self.fuser = nn.Linear(3, 2)
        self.heads = nn.ModuleDict({"map": nn.ModuleDict({"classifier": nn.ModuleList([
            nn.Identity(), nn.Identity(), nn.Identity(), nn.Identity(),
            nn.Identity(), nn.Identity(), nn.Linear(2, 6),
        ])})})


def test_loader_excludes_semantic_output_even_when_shape_matches(tmp_path):
    source = TinyFusion()
    with torch.no_grad():
        source.encoders["camera"]["backbone"].weight.fill_(3)
        source.heads["map"]["classifier"][6].weight.fill_(9)
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": source.state_dict()}, checkpoint)
    target = TinyFusion()
    old_head = target.heads["map"]["classifier"][6].weight.detach().clone()
    report = load_selective_checkpoint(
        target,
        checkpoint,
        exclude_prefixes=("heads.map.classifier.6",),
        required_prefixes=("encoders.camera.backbone",),
    )
    assert torch.all(target.encoders["camera"]["backbone"].weight == 3)
    torch.testing.assert_close(target.heads["map"]["classifier"][6].weight, old_head)
    assert "heads.map.classifier.6.weight" in report.excluded


def test_loader_skips_shape_mismatch_and_requires_core_prefix(tmp_path):
    source = TinyFusion()
    state = source.state_dict()
    state["fuser.weight"] = torch.ones((9, 9))
    checkpoint = tmp_path / "model.pth"
    torch.save({"state_dict": state}, checkpoint)
    report = load_selective_checkpoint(
        TinyFusion(), checkpoint,
        exclude_prefixes=("heads.map.classifier.6",),
        required_prefixes=("encoders.camera.backbone",),
    )
    assert "fuser.weight" in report.shape_mismatch
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `pytest -q tests/test_robot_bev_checkpoint.py`

Expected: FAIL with `No module named 'mmdet3d.utils.checkpoint'`.

- [ ] **Step 3: Implement report and filter**

```python
@dataclass
class SelectiveLoadReport:
    loaded: List[str]
    excluded: List[str]
    shape_mismatch: List[str]
    missing_in_model: List[str]


def load_selective_checkpoint(
    model,
    filename,
    exclude_prefixes,
    required_prefixes,
    map_location="cpu",
):
    module = model.module if hasattr(model, "module") else model
    checkpoint = torch.load(str(filename), map_location=map_location)
    source = checkpoint.get("state_dict", checkpoint)
    source = {_strip_module_prefix(key): value for key, value in source.items()}
    target = module.state_dict()
    loaded, excluded, mismatched, missing = [], [], [], []
    filtered = {}
    geometry_suffixes = (".dx", ".bx", ".nx", ".frustum")
    for key, value in source.items():
        if key.startswith(tuple(exclude_prefixes)) or key.endswith(geometry_suffixes):
            excluded.append(key)
        elif key not in target:
            missing.append(key)
        elif tuple(value.shape) != tuple(target[key].shape):
            mismatched.append(key)
        else:
            filtered[key] = value
            loaded.append(key)
    absent_required = [
        prefix for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in loaded)
    ]
    if absent_required:
        raise RuntimeError(f"checkpoint did not load required modules: {absent_required}")
    module.load_state_dict(filtered, strict=False)
    report = SelectiveLoadReport(loaded, excluded, mismatched, missing)
    _log_report(report)
    return report
```

Implement the Python-3.8-compatible helper exactly as follows; `_log_report()` reports counts and groups names by top-level module without printing full tensor contents.

```python
def _strip_module_prefix(key):
    return key[len("module."):] if key.startswith("module.") else key
```

- [ ] **Step 4: Integrate train precedence**

Export the loader from `mmdet3d/utils/__init__.py`. In `train_model()` use:

```python
if cfg.resume_from:
    runner.resume(cfg.resume_from)
elif cfg.get("selective_load_from"):
    selective_cfg = cfg.get("selective_load", {})
    load_selective_checkpoint(
        runner.model,
        cfg.selective_load_from,
        exclude_prefixes=tuple(selective_cfg.get("exclude_prefixes", ())),
        required_prefixes=tuple(selective_cfg.get("required_prefixes", ())),
    )
elif cfg.load_from:
    runner.load_checkpoint(cfg.load_from)
```

- [ ] **Step 5: Run checkpoint tests and commit**

Run: `pytest -q tests/test_robot_bev_checkpoint.py`

Expected: all tests PASS under Python 3.8.

```bash
git add mmdet3d/utils/checkpoint.py mmdet3d/utils/__init__.py \
  mmdet3d/apis/train.py tests/test_robot_bev_checkpoint.py
git commit -m "feat: selectively load robot BEV pretrained weights"
```

---

### Task 6: Dedicated Robot BEV training configuration

**Files:**
- Create: `configs/robot_bev/default.yaml`
- Create: `configs/robot_bev/seg/camera_lidar_lss.yaml`
- Test: `tests/test_robot_bev_config.py`

**Interfaces:**
- Consumes: derived BEVFusion indexes, new pipeline transform, selective checkpoint loader.
- Produces: a recursively loadable camera+LiDAR LSS segmentation experiment with masked supervision and no BEV-desynchronizing augmentation.

- [ ] **Step 1: Write resolved-config assertions**

```python
# tests/test_robot_bev_config.py
from mmcv import Config
from torchpack.utils.config import configs

from mmdet3d.utils import recursive_eval


def test_robot_bev_config_contract():
    configs.load("configs/robot_bev/seg/camera_lidar_lss.yaml", recursive=True)
    cfg = Config(recursive_eval(configs))
    assert cfg.map_classes == [
        "floor", "carpet", "obstacle", "wall", "furniture", "other"
    ]
    assert cfg.data.train.type == "NuScenesDataset"
    assert cfg.data.train.resolve_relative_paths is True
    assert cfg.model.heads.map.classes == cfg.map_classes
    assert cfg.model.heads.object is None
    pipeline_types = [step.type for step in cfg.data.train.pipeline]
    assert "LoadRobotBEVSegmentation" in pipeline_types
    assert "RandomFlip3D" not in pipeline_types
    assert "GTDepth" not in pipeline_types
    collect = next(step for step in cfg.data.train.pipeline if step.type == "Collect3D")
    assert "gt_supervision_mask_bev" in collect.keys
    assert cfg.selective_load.exclude_prefixes == ["heads.map.classifier.6"]
```

- [ ] **Step 2: Run test and verify missing config**

Run: `pytest -q tests/test_robot_bev_config.py`

Expected: FAIL because `configs/robot_bev/seg/camera_lidar_lss.yaml` does not exist.

- [ ] **Step 3: Create dataset and pipeline defaults**

`configs/robot_bev/default.yaml` must set:

```yaml
dataset_type: NuScenesDataset
dataset_root: data/replica_robot_bev_v3/
map_classes: [floor, carpet, obstacle, wall, furniture, other]
object_classes: []
point_cloud_range: [0.0, -1.52, -0.5, 3.04, 1.52, 2.5]
voxel_size: [0.005, 0.005, 0.1]
image_size: [256, 704]
input_modality:
  use_lidar: true
  use_camera: true
  use_radar: false
  use_map: false
  use_external: false
```

Define train and test pipelines using, in order: image load, current points, five sweeps, image augmentation, identity `GlobalRotScaleTrans`, `LoadRobotBEVSegmentation`, point range filter, image normalize, optional train-only GridMask with probability 0 initially, point shuffle for train, `DefaultFormatBundle3D`, and `Collect3D`. Set `LoadPointsFromMultiSweeps.sweeps_num: 5` and `pad_empty_sweeps: false`, so early scene frames use only history that actually exists. Collect `img`, `points`, `gt_masks_bev`, `gt_supervision_mask_bev` plus all calibration/augmentation meta keys. Do not include detection annotations, object filters, RandomFlip3D, or GTDepth.

Define direct NuScenesDataset train/val/test entries with `resolve_relative_paths: true`, `filter_empty_gt: false`, and corresponding `bevfusion_infos_<split>.pkl`.

- [ ] **Step 4: Create the exact model/fine-tuning config**

Start from the existing robot camera+LiDAR model block, with these required values:

```yaml
model:
  type: BEVFusion
  encoders:
    camera:
      vtransform:
        type: LSSTransform
        in_channels: 256
        out_channels: 80
        image_size: ${image_size}
        feature_size: ${[image_size[0] // 8, image_size[1] // 8]}
        xbound: [0.0, 3.04, 0.02]
        ybound: [-1.52, 1.52, 0.02]
        zbound: [-0.5, 2.5, 3.0]
        dbound: [0.1, 5.0, 0.05]
        downsample: 2
    lidar:
      voxelize:
        max_num_points: 10
        point_cloud_range: ${point_cloud_range}
        voxel_size: ${voxel_size}
        max_voxels: [60000, 90000]
  fuser:
    type: ConvFuser
    in_channels: [80, 128]
    out_channels: 256
  heads:
    object: null
    map:
      type: BEVSegmentationHead
      in_channels: 512
      grid_transform:
        input_scope: [[0.0, 3.04, 0.04], [-1.52, 1.52, 0.04]]
        output_scope: [[0.0, 3.0, 0.02], [-1.5, 1.5, 0.02]]
      classes: ${map_classes}
      loss: focal
```

Copy the camera backbone/neck, complete SparseEncoder, SECOND decoder backbone, and SECONDFPN decoder neck blocks from `configs/nuscenes/seg/robot-fusion-bev150-lss.yaml` without changing channel definitions.

Add:

```yaml
load_from: null
resume_from: null
selective_load_from: pretrained/bevfusion-seg.pth
selective_load:
  exclude_prefixes: [heads.map.classifier.6]
  required_prefixes:
    - encoders.camera.backbone
    - encoders.camera.neck
    - encoders.lidar.backbone
    - decoder.backbone
    - decoder.neck
optimizer:
  type: AdamW
  lr: 1.0e-4
  weight_decay: 0.01
  paramwise_cfg:
    custom_keys:
      encoders.camera.backbone:
        lr_mult: 0.1
      encoders.camera.neck:
        lr_mult: 0.5
      absolute_pos_embed:
        decay_mult: 0
      relative_position_bias_table:
        decay_mult: 0
evaluation:
  interval: 1
  metric: map
  save_best: map/mean/iou@0.50
  rule: greater
```

- [ ] **Step 5: Run config test and build dataset/model in the BEVFusion environment**

Run: `pytest -q tests/test_robot_bev_config.py`

Expected: PASS.

Run with an existing converted smoke root:

```bash
python -c "from torchpack.utils.config import configs; from mmcv import Config; from mmdet3d.utils import recursive_eval; from mmdet3d.datasets import build_dataset; from mmdet3d.models import build_model; configs.load('configs/robot_bev/seg/camera_lidar_lss.yaml', recursive=True); cfg=Config(recursive_eval(configs)); print(len(build_dataset(cfg.data.train))); print(type(build_model(cfg.model)).__name__)"
```

Expected: dataset length equals the converted smoke train count; model type is `BEVFusion`.

- [ ] **Step 6: Commit the canonical training config**

```bash
git add configs/robot_bev/default.yaml \
  configs/robot_bev/seg/camera_lidar_lss.yaml tests/test_robot_bev_config.py
git commit -m "feat: add robot BEV segmentation config"
```

---

### Task 7: End-to-end smoke, overfit gate, remote runbook, and regression suite

**Files:**
- Create: `tests/test_robot_bev_integration.py`
- Create: `docs/robot_bev_training.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: canonical smoke dataset, converted indexes, dedicated config, pretrained checkpoint.
- Produces: one-batch finite forward/backward proof, small-overfit commands, full remote commands, final-test commands, and top-level discoverability.

- [ ] **Step 1: Write opt-in integration test**

```python
# tests/test_robot_bev_integration.py
import os

import pytest
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


@pytest.mark.skipif(
    "ROBOT_BEV_DATA_ROOT" not in os.environ,
    reason="set ROBOT_BEV_DATA_ROOT to a validated converted smoke dataset",
)
def test_one_batch_forward_backward_is_finite():
    configs.load("configs/robot_bev/seg/camera_lidar_lss.yaml", recursive=True)
    cfg = Config(recursive_eval(configs))
    root = os.environ["ROBOT_BEV_DATA_ROOT"].rstrip("/") + "/"
    cfg.data.train.dataset_root = root
    cfg.data.train.ann_file = root + "bevfusion_infos_train.pkl"
    dataset = build_dataset(cfg.data.train)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
    )
    model = MMDataParallel(build_model(cfg.model).cuda().train(), device_ids=[0])
    batch = next(iter(loader))
    losses = model(**batch)
    total = sum(value.mean() for key, value in losses.items() if "loss" in key)
    assert torch.isfinite(total)
    total.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
```

- [ ] **Step 2: Run dependency-level regression tests**

Run:

```bash
pytest -q \
  tests/test_robot_bev_schema.py \
  tests/test_robot_bev_writer.py \
  tests/test_robot_bev_validator.py \
  tests/test_robot_bev_geometry.py \
  tests/test_robot_bev_converter.py \
  tests/test_robot_bev_pipeline.py \
  tests/test_bev_valid_mask.py \
  tests/test_robot_bev_checkpoint.py \
  tests/test_robot_bev_config.py
```

Expected: all non-Habitat tests PASS.

- [ ] **Step 3: Run the one-batch gate on the regenerated smoke dataset**

```bash
export ROBOT_BEV_DATA_ROOT=/data/replica_robot_bev_v3_smoke
pytest -q tests/test_robot_bev_integration.py -s
```

Expected: PASS with finite loss and gradients.

- [ ] **Step 4: Run the 16–32-frame overfit gate**

Generate a separate canonical 32-frame training smoke root with the Plan-1 generator, validate it, and convert it. Then run:

```bash
CUDA_VISIBLE_DEVICES=0 torchpack dist-run -np 1 python tools/train.py \
  configs/robot_bev/seg/camera_lidar_lss.yaml \
  --run-dir runs/robot-bev-overfit \
  --data.samples_per_gpu 1 \
  --data.workers_per_gpu 0 \
  --max_epochs 50
```

Expected: total/map losses decrease clearly from their initial values; predictions remain aligned with GT in camera and BEV visualizations. Record initial/final loss and fixed `mIoU@0.50` in the runbook verification table.

- [ ] **Step 5: Write exact remote generate/validate/convert/train/evaluate commands**

`docs/robot_bev_training.md` must include:

```bash
python -m data_generation.robot_bev.cli.validate_dataset \
  --root /data/replica_robot_bev_v3 \
  --geometry-scene office_1 --geometry-frame 0

python tools/data_converter/robot_bev_converter.py \
  --root /data/replica_robot_bev_v3 --split all --max-sweeps 5

CUDA_VISIBLE_DEVICES=0,1,2,3 torchpack dist-run -np 4 python tools/train.py \
  configs/robot_bev/seg/camera_lidar_lss.yaml \
  --run-dir runs/robot-bev-replica-v3 \
  --dataset_root /data/replica_robot_bev_v3/

CUDA_VISIBLE_DEVICES=0 torchpack dist-run -np 1 python tools/test.py \
  configs/robot_bev/seg/camera_lidar_lss.yaml \
  runs/robot-bev-replica-v3/latest.pth \
  --eval map \
  --cfg-options \
    data.test.dataset_root=/data/replica_robot_bev_v3/ \
    data.test.ann_file=/data/replica_robot_bev_v3/bevfusion_infos_test.pkl
```

Verify the exact torchpack override syntax against `tools/train.py`; if root-level `--dataset_root` does not update nested evaluated paths, document explicit `--data.train.dataset_root`, `--data.train.ann_file`, `--data.val.dataset_root`, and `--data.val.ann_file` overrides instead. Include resume command, selective-pretrain report expectations, GPU batch-size adjustment, fixed-metric model selection, a single final test invocation, and multi-root ConcatDataset example.

- [ ] **Step 6: Add top-level documentation link and run final checks**

Add a concise `Robot BEV data generation and training` section to `README.md` linking the toolkit README, schema spec, and training runbook.

Run: `git diff --check`

Expected: no output.

Run the complete regression suite and opt-in integration test again. Expected: all PASS.

- [ ] **Step 7: Commit end-to-end workflow**

```bash
git add tests/test_robot_bev_integration.py docs/robot_bev_training.md README.md
git commit -m "docs: add robot BEV training workflow"
```

## Plan 2 Completion Gate

Implementation is complete only when the full unit suite passes, the regenerated smoke dataset passes validation and human geometry review, one batch has finite loss/gradients, 16–32 frames visibly overfit, selective loading reports the expected reused/skipped modules, and the remote runbook contains the exact commands used. Full 18×600 generation and full remote training remain operator-run activities after these gates.
