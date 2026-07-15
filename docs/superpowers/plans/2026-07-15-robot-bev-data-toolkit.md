# Robot BEV Data Toolkit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tracked, source-independent Robot BEV data toolkit and refactor the Replica/Habitat-Sim generator to emit validated canonical schema-v3 datasets.

**Architecture:** This is plan 1 of 2. A dependency-light schema/writer/validator layer owns the canonical dataset contract; Habitat-Sim common rendering and Replica-specific behavior live behind source adapters. The generator writes framework-neutral records only, while geometry diagnostics and strict validation gate every completed dataset.

**Tech Stack:** Python 3.8, NumPy, Pillow, pytest, Habitat-Sim 0.2.2 for Replica rendering, JSON/JSONL, pickle.

## Global Constraints

- The canonical identity is exactly `schema_name=robot_bev_dataset` and `schema_version=3`.
- Classes are exactly `floor`, `carpet`, `obstacle`, `wall`, `furniture`, `other` in that order.
- BEV bounds are exactly `xbound=[0.0,3.0,0.02]` and `ybound=[-1.5,1.5,0.02]`; semantic masks are `uint8 [6,150,150]`.
- Base/LiDAR axes are x-forward, y-left, z-up; camera axes are OpenCV x-right, y-down, z-forward; transforms use column vectors.
- Point files are packed `float32 [x,y,z,intensity,time]` in meters.
- Every indexed path is POSIX and relative to the explicit dataset root; absolute paths and `..` are rejected.
- `bev_observed_mask` is required; `class_validity [6]` is required; a per-class supervision mask is optional.
- Current Replica frames use `class_validity=[1,1,1,1,1,1]` and do not write duplicated per-class supervision masks.
- Splits are scene-level and mutually exclusive; test scenes never participate in model selection.
- Replica production rendering requires Habitat-Sim exactly 0.2.2.
- The generator writes no NuScenes/MMDetection3D fields and no precomputed sweeps.
- Existing 90-frame and remote 18×600-frame outputs are regenerated; no migration utility is built.
- Do not import Habitat-Sim from schema, writer, validator, or geometry unit tests.
- Execute toolkit unit tests in a Python 3.8 environment containing NumPy, Pillow, and pytest; execute Replica integration commands in the `habitat022` environment.

## File Map

- `data_generation/__init__.py`: tracked top-level package marker.
- `data_generation/robot_bev/schema.py`: constants, canonical tokens, relative-path checks, effective supervision-mask composition.
- `data_generation/robot_bev/writer.py`: frame payload API, atomic files, manifests, indexes, fingerprints, final summaries.
- `data_generation/robot_bev/validator.py`: structural/numeric validation and machine-readable reports.
- `data_generation/robot_bev/geometry_checks.py`: projection, BEV indexing, sweep alignment, diagnostic images.
- `data_generation/robot_bev/sources/habitat_common.py`: Habitat sensor/pose/navigation/depth helpers shared by future adapters.
- `data_generation/robot_bev/sources/replica.py`: Replica PTex/NavMesh/semantic mapping and frame-generation loop.
- `data_generation/robot_bev/cli/generate_replica.py`: Replica generation CLI.
- `data_generation/robot_bev/cli/validate_dataset.py`: validation/geometry CLI.
- `data_generation/robot_bev/configs/*`: tracked scene and split examples.
- `data_generation/robot_bev/docs/*`: schema, Replica operation, adapter, and quality guides.
- `tests/test_robot_bev_*.py`: dependency-light unit/integration tests.

---

### Task 1: Canonical schema and supervision-mask composition

**Files:**
- Create: `data_generation/__init__.py`
- Create: `data_generation/robot_bev/__init__.py`
- Create: `data_generation/robot_bev/schema.py`
- Test: `tests/test_robot_bev_schema.py`

**Interfaces:**
- Consumes: NumPy arrays and user-supplied relative paths.
- Produces: `MAP_CLASSES`, `BEV_XBOUND`, `BEV_YBOUND`, `canonical_token()`, `normalize_relative_path()`, `effective_supervision_mask()`, and `SchemaError`.

- [ ] **Step 1: Write failing schema tests**

```python
# tests/test_robot_bev_schema.py
from pathlib import Path

import numpy as np
import pytest

from data_generation.robot_bev.schema import (
    BEV_SHAPE,
    MAP_CLASSES,
    SchemaError,
    canonical_token,
    effective_supervision_mask,
    normalize_relative_path,
)


def test_schema_constants_are_fixed():
    assert MAP_CLASSES == (
        "floor",
        "carpet",
        "obstacle",
        "wall",
        "furniture",
        "other",
    )
    assert BEV_SHAPE == (6, 150, 150)


def test_relative_path_is_portable_and_cannot_escape_root():
    assert normalize_relative_path(Path("office_0/images/000012.png")) == (
        "office_0/images/000012.png"
    )
    assert normalize_relative_path("office_0\\images\\000012.png") == (
        "office_0/images/000012.png"
    )
    for invalid in ("/tmp/frame.png", "../frame.png", "C:/frame.png", ""):
        with pytest.raises(SchemaError):
            normalize_relative_path(invalid)


def test_effective_mask_broadcasts_without_duplicate_storage():
    observed = np.zeros((150, 150), dtype=np.uint8)
    observed[10:20, 30:40] = 1
    class_validity = np.array([1, 0, 1, 1, 1, 1], dtype=np.uint8)
    effective = effective_supervision_mask(observed, class_validity)
    assert effective.shape == (6, 150, 150)
    assert effective.dtype == np.uint8
    assert effective[0].sum() == 100
    assert effective[1].sum() == 0


def test_optional_per_class_mask_is_intersected():
    observed = np.ones((150, 150), dtype=np.uint8)
    class_validity = np.ones((6,), dtype=np.uint8)
    regional = np.ones((6, 150, 150), dtype=np.uint8)
    regional[4, :, 75:] = 0
    effective = effective_supervision_mask(observed, class_validity, regional)
    assert effective[4, :, 75:].sum() == 0
    assert effective[4, :, :75].sum() == 150 * 75


def test_token_is_dataset_scene_and_frame_scoped():
    assert canonical_token("replica_v3", "office_0", 12) == (
        "replica_v3:office_0:000012"
    )
```

- [ ] **Step 2: Run the tests and verify import failure**

Run: `pytest -q tests/test_robot_bev_schema.py`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'data_generation'`.

- [ ] **Step 3: Implement the canonical schema API**

```python
# data_generation/robot_bev/schema.py
from pathlib import PurePosixPath
from typing import Optional, Sequence, Tuple, Union

import numpy as np

SCHEMA_NAME = "robot_bev_dataset"
SCHEMA_VERSION = 3
MAP_CLASSES: Tuple[str, ...] = (
    "floor",
    "carpet",
    "obstacle",
    "wall",
    "furniture",
    "other",
)
BEV_XBOUND = (0.0, 3.0, 0.02)
BEV_YBOUND = (-1.5, 1.5, 0.02)
BEV_SHAPE = (6, 150, 150)
OBSERVED_MASK_SHAPE = (150, 150)
POINT_DIMENSIONS = ("x", "y", "z", "intensity", "time")


class SchemaError(ValueError):
    pass


def canonical_token(dataset_id: str, scene_id: str, frame_id: int) -> str:
    if not dataset_id or not scene_id or frame_id < 0:
        raise SchemaError("dataset_id and scene_id are required; frame_id must be non-negative")
    if ":" in dataset_id or ":" in scene_id:
        raise SchemaError("dataset_id and scene_id must not contain ':'")
    return f"{dataset_id}:{scene_id}:{frame_id:06d}"


def normalize_relative_path(path: Union[str, PurePosixPath]) -> str:
    text = str(path).replace("\\", "/")
    candidate = PurePosixPath(text)
    if not text or candidate.is_absolute() or ".." in candidate.parts:
        raise SchemaError(f"path must remain inside the dataset root: {path!r}")
    if candidate.parts and candidate.parts[0].endswith(":"):
        raise SchemaError(f"Windows absolute paths are forbidden: {path!r}")
    normalized = candidate.as_posix()
    if normalized in ("", "."):
        raise SchemaError(f"empty relative path is forbidden: {path!r}")
    return normalized


def _binary_uint8(value: np.ndarray, shape: Sequence[int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != tuple(shape):
        raise SchemaError(f"{name} shape {array.shape} != {tuple(shape)}")
    if array.dtype != np.uint8:
        raise SchemaError(f"{name} dtype {array.dtype} != uint8")
    if not np.isin(array, (0, 1)).all():
        raise SchemaError(f"{name} must be binary")
    return array


def effective_supervision_mask(
    observed_mask: np.ndarray,
    class_validity: np.ndarray,
    per_class_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    observed = _binary_uint8(observed_mask, OBSERVED_MASK_SHAPE, "observed_mask")
    validity = _binary_uint8(class_validity, (len(MAP_CLASSES),), "class_validity")
    effective = observed[None, :, :] * validity[:, None, None]
    if per_class_mask is not None:
        regional = _binary_uint8(per_class_mask, BEV_SHAPE, "per_class_mask")
        effective = effective * regional
    return effective.astype(np.uint8, copy=False)
```

Create empty package markers in `data_generation/__init__.py` and `data_generation/robot_bev/__init__.py`, then re-export the schema constants and functions from the latter.

- [ ] **Step 4: Run schema tests**

Run: `pytest -q tests/test_robot_bev_schema.py`

Expected: `5 passed`.

- [ ] **Step 5: Commit the schema contract**

```bash
git add data_generation/__init__.py data_generation/robot_bev/__init__.py \
  data_generation/robot_bev/schema.py tests/test_robot_bev_schema.py
git commit -m "feat: add canonical robot BEV schema"
```

---

### Task 2: Atomic dataset writer, manifests, indexes, and resume fingerprints

**Files:**
- Create: `data_generation/robot_bev/writer.py`
- Test: `tests/test_robot_bev_writer.py`

**Interfaces:**
- Consumes: `FramePayload`, root metadata, scene-level splits, and source-generation parameters.
- Produces: `RobotBEVWriter.write_frame()`, `finalize_scene()`, `finalize_dataset()`, root-relative manifests/indexes, and deterministic fingerprints.

- [ ] **Step 1: Write failing writer tests**

```python
# tests/test_robot_bev_writer.py
import json
import pickle

import numpy as np

from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def make_payload(frame_id: int) -> FramePayload:
    return FramePayload(
        frame_id=frame_id,
        timestamp=1_000_000 + frame_id * 100_000,
        rgb=np.zeros((8, 12, 3), dtype=np.uint8),
        points=np.array([[1.0, 0.0, 0.1, 0.0, 0.0]], dtype=np.float32),
        bev_labels=np.zeros((6, 150, 150), dtype=np.uint8),
        observed_mask=np.ones((150, 150), dtype=np.uint8),
        class_validity=np.ones((6,), dtype=np.uint8),
        cam_intrinsic=np.eye(3, dtype=np.float32),
        camera2base=np.eye(4, dtype=np.float32),
        lidar2base=np.eye(4, dtype=np.float32),
        map_from_base=np.eye(4, dtype=np.float32),
    )


def test_writer_creates_root_relative_canonical_indexes(tmp_path):
    writer = RobotBEVWriter(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"width": 12, "height": 8},
    )
    writer.write_frame("scene_a", "train", make_payload(0))
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()

    with (tmp_path / "robot_infos_train.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    info = payload["infos"][0]
    assert info["image_path"] == "scene_a/images/000000.png"
    assert info["token"] == "fixture_v3:scene_a:000000"
    assert info["class_validity"].tolist() == [1, 1, 1, 1, 1, 1]
    assert "sweeps" not in info
    metadata = json.loads((tmp_path / "dataset_metadata.json").read_text())
    assert metadata["schema_version"] == 3


def test_writer_refuses_resume_when_generation_contract_changes(tmp_path):
    common = dict(
        root=tmp_path,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
    )
    RobotBEVWriter(generation_parameters={"hfov": 120.0}, **common)
    try:
        RobotBEVWriter(
            generation_parameters={"hfov": 90.0}, resume=True, **common
        )
    except RuntimeError as error:
        assert "generation fingerprint mismatch" in str(error)
    else:
        raise AssertionError("resume must reject changed generation parameters")
```

- [ ] **Step 2: Run writer tests and verify missing module**

Run: `pytest -q tests/test_robot_bev_writer.py`

Expected: FAIL during collection with `No module named 'data_generation.robot_bev.writer'`.

- [ ] **Step 3: Implement the writer public API and atomic helpers**

Implement these exact public types and methods in `writer.py`:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FramePayload:
    frame_id: int
    timestamp: int
    rgb: np.ndarray
    points: np.ndarray
    bev_labels: np.ndarray
    observed_mask: np.ndarray
    class_validity: np.ndarray
    cam_intrinsic: np.ndarray
    camera2base: np.ndarray
    lidar2base: np.ndarray
    map_from_base: np.ndarray
    per_class_supervision_mask: Optional[np.ndarray] = None
    depth_mm: Optional[np.ndarray] = None
    semantics: Optional[np.ndarray] = None


class RobotBEVWriter:
    def __init__(
        self,
        root: Path,
        dataset_id: str,
        source_type: str,
        source_dataset: str,
        generator_name: str,
        generator_version: str,
        splits: Mapping[str, Sequence[str]],
        generation_parameters: Mapping[str, object],
        resume: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.dataset_id = dataset_id
        self.splits = {name: list(splits.get(name, [])) for name in ("train", "val", "test")}
        self.generation_parameters = dict(generation_parameters)
        self.metadata = self._build_metadata(
            source_type, source_dataset, generator_name, generator_version
        )
        self._validate_splits()
        self._initialize_root(resume)

    def write_frame(self, scene_id: str, split: str, frame: FramePayload) -> Dict[str, object]:
        self._validate_frame(scene_id, split, frame)
        record = self._write_frame_artifacts(scene_id, frame)
        self._append_manifest(scene_id, record)
        return record

    def finalize_scene(self, scene_id: str, split: str) -> Dict[str, object]:
        records = self._load_manifest(scene_id)
        infos = [self._manifest_to_info(scene_id, index, record) for index, record in enumerate(records)]
        self._atomic_pickle(self.root / scene_id / "scene_infos.pkl", self._index_payload(infos))
        self._atomic_pickle(self.root / scene_id / f"robot_infos_{split}.pkl", self._index_payload(infos))
        summary = self._scene_summary(scene_id, split, records)
        self._atomic_json(self.root / scene_id / "summary.json", summary)
        return summary

    def finalize_dataset(self) -> Dict[str, object]:
        split_infos = {name: [] for name in ("train", "val", "test")}
        summaries: List[Dict[str, object]] = []
        for split, scene_ids in self.splits.items():
            for scene_id in scene_ids:
                scene_payload = self._read_pickle(self.root / scene_id / "scene_infos.pkl")
                split_infos[split].extend(scene_payload["infos"])
                summaries.append(self._read_json(self.root / scene_id / "summary.json"))
        for split, infos in split_infos.items():
            self._atomic_pickle(self.root / f"robot_infos_{split}.pkl", self._index_payload(infos))
        root_summary = {
            "status": "complete",
            "info_counts": {name: len(infos) for name, infos in split_infos.items()},
            "scene_summaries": summaries,
            "failures": [],
        }
        self._atomic_json(self.root / "multi_scene_summary.json", root_summary)
        return root_summary
```

Implement the named private methods with these exact rules:

- `_build_metadata()` writes every field from design section 5 plus a SHA-256 fingerprint of sorted compact JSON containing schema constants and `generation_parameters`.
- `_validate_splits()` rejects unknown split names, duplicates, empty scene IDs, and cross-split overlap.
- `_initialize_root()` creates root metadata and `splits.json` for a new root; with `resume=True`, it reads them and rejects any fingerprint or split mismatch.
- `_validate_frame()` uses schema shape/binary checks, requires finite matrices/points, requires point shape `[N,5]`, requires labels outside observed mask to be zero, and requires contiguous frame IDs in each manifest.
- `_write_frame_artifacts()` atomically writes RGB PNG, point BIN, BEV label NPY, observed NPY, and optional depth/semantic/supervision files; every returned path is root-relative.
- `_append_manifest()` rewrites the complete JSONL through a temporary file followed by `os.replace`, so a crash cannot leave a partial last line.
- `_manifest_to_info()` emits section-7 fields, derives canonical token/prev_token, and stores matrices as `float32` arrays.
- `_atomic_json()`, `_atomic_pickle()`, `_atomic_npy()`, `_atomic_png()`, and `_atomic_points()` create a sibling `.tmp` file, flush and `os.fsync`, then call `os.replace`.
- `_scene_summary()` reports frame count, point min/max/mean, per-class sums, observed sum, and split.
- `_index_payload()` copies root metadata and adds `scene_split` only for per-scene indexes.

- [ ] **Step 4: Run writer tests**

Run: `pytest -q tests/test_robot_bev_schema.py tests/test_robot_bev_writer.py`

Expected: `7 passed`.

- [ ] **Step 5: Commit atomic writer support**

```bash
git add data_generation/robot_bev/writer.py tests/test_robot_bev_writer.py
git commit -m "feat: add canonical robot BEV writer"
```

---

### Task 3: Strict structural and numeric validator

**Files:**
- Create: `data_generation/robot_bev/validator.py`
- Test: `tests/test_robot_bev_validator.py`

**Interfaces:**
- Consumes: a canonical dataset root and optional split.
- Produces: `validate_dataset(root, split=None) -> ValidationReport`; raises `DatasetValidationError` with dataset/scene/frame/field context.

- [ ] **Step 1: Write failing validator tests using the writer fixture**

```python
# tests/test_robot_bev_validator.py
import json

import numpy as np
import pytest

from data_generation.robot_bev.validator import DatasetValidationError, validate_dataset
from data_generation.robot_bev.writer import FramePayload, RobotBEVWriter


def build_dataset(root):
    writer = RobotBEVWriter(
        root=root,
        dataset_id="fixture_v3",
        source_type="simulation",
        source_dataset="fixture",
        generator_name="pytest",
        generator_version="1",
        splits={"train": ["scene_a"], "val": [], "test": []},
        generation_parameters={"fixture": True},
    )
    frame = FramePayload(
        frame_id=0,
        timestamp=1_000_000,
        rgb=np.zeros((8, 12, 3), dtype=np.uint8),
        points=np.zeros((1, 5), dtype=np.float32),
        bev_labels=np.zeros((6, 150, 150), dtype=np.uint8),
        observed_mask=np.ones((150, 150), dtype=np.uint8),
        class_validity=np.ones((6,), dtype=np.uint8),
        cam_intrinsic=np.array([[10, 0, 6], [0, 10, 4], [0, 0, 1]], dtype=np.float32),
        camera2base=np.eye(4, dtype=np.float32),
        lidar2base=np.eye(4, dtype=np.float32),
        map_from_base=np.eye(4, dtype=np.float32),
    )
    writer.write_frame("scene_a", "train", frame)
    writer.finalize_scene("scene_a", "train")
    writer.finalize_dataset()


def test_validator_accepts_a_complete_dataset(tmp_path):
    build_dataset(tmp_path)
    report = validate_dataset(tmp_path)
    assert report.valid
    assert report.frame_counts == {"train": 1, "val": 0, "test": 0}


def test_validator_reports_context_for_path_escape(tmp_path):
    build_dataset(tmp_path)
    manifest_path = tmp_path / "scene_a" / "manifest.jsonl"
    record = json.loads(manifest_path.read_text().strip())
    record["image_path"] = "../escape.png"
    manifest_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(DatasetValidationError) as caught:
        validate_dataset(tmp_path)
    message = str(caught.value)
    assert "fixture_v3" in message
    assert "scene_a" in message
    assert "image_path" in message
```

- [ ] **Step 2: Run validator tests and verify import failure**

Run: `pytest -q tests/test_robot_bev_validator.py`

Expected: FAIL with `No module named 'data_generation.robot_bev.validator'`.

- [ ] **Step 3: Implement validator report and checks**

```python
# Public API in data_generation/robot_bev/validator.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


class DatasetValidationError(RuntimeError):
    def __init__(self, dataset_id, scene_id, frame_id, field_name, message):
        context = (
            f"dataset={dataset_id} scene={scene_id} frame={frame_id} "
            f"field={field_name}"
        )
        super().__init__(f"{context}: {message}")


@dataclass
class ValidationReport:
    valid: bool
    frame_counts: Dict[str, int]
    warnings: List[str] = field(default_factory=list)


def validate_dataset(root: Path, split: Optional[str] = None) -> ValidationReport:
    root = Path(root).expanduser().resolve()
    metadata = _load_root_metadata(root)
    splits = _load_and_validate_splits(root, metadata)
    selected = (split,) if split is not None else ("train", "val", "test")
    counts = {name: 0 for name in ("train", "val", "test")}
    warnings: List[str] = []
    seen_tokens = set()
    for split_name in selected:
        infos = _load_index(root, split_name, metadata)
        expected_scenes = set(splits[split_name])
        _validate_manifests_match_index(
            root, metadata, split_name, expected_scenes, infos
        )
        previous_by_scene = {}
        for info in infos:
            _validate_frame_info(root, metadata, split_name, expected_scenes, info)
            token = info["token"]
            if token in seen_tokens:
                _fail(metadata, info, "token", f"duplicate token {token}")
            seen_tokens.add(token)
            _validate_sequence(previous_by_scene, metadata, info)
            _validate_artifacts(root, metadata, info)
            counts[split_name] += 1
        warnings.extend(_quality_warnings(root, metadata, split_name, infos))
    return ValidationReport(valid=True, frame_counts=counts, warnings=warnings)
```

Implement the private checks explicitly listed in design section 13: supported schema/version, exact class order and BEV metadata, disjoint splits, root-contained paths, files present, consistent frame counts, token/timestamp/prev continuity, mask shape/dtype/binary rules, labels zero outside observed, optional supervision shape, point byte divisibility and finite values, valid intrinsics, homogeneous bottom rows, orthonormal rotations with determinant within `1e-3` of `+1`, and `pose_valid is True`.

`_validate_manifests_match_index()` must load every expected scene manifest, validate its JSON records, convert scene-relative artifact paths to root-relative paths, and compare frame ID, timestamp, artifact paths, and pose against the root index. This ensures tampering or interrupted index generation is detected in either representation.

Quality warnings are returned for zero class totals, observed coverage below 1%, point counts outside five times the split median, and intensity outside `[0,1]`; they never rewrite data.

- [ ] **Step 4: Add corruption cases and run validator tests**

Add parametrized tests that corrupt mask dtype, mask shape, rotation determinant, point byte length, token uniqueness, and split overlap. Each case must assert `DatasetValidationError` names the affected field.

Run: `pytest -q tests/test_robot_bev_schema.py tests/test_robot_bev_writer.py tests/test_robot_bev_validator.py`

Expected: all tests PASS.

- [ ] **Step 5: Commit the validation gate**

```bash
git add data_generation/robot_bev/validator.py tests/test_robot_bev_validator.py
git commit -m "feat: validate canonical robot BEV datasets"
```

---

### Task 4: Geometry diagnostics for camera, BEV, and sweeps

**Files:**
- Create: `data_generation/robot_bev/geometry_checks.py`
- Test: `tests/test_robot_bev_geometry.py`

**Interfaces:**
- Consumes: canonical frame matrices, points, images, and two poses.
- Produces: `project_lidar_to_image()`, `points_to_bev_cells()`, `history_to_current_lidar()`, and `write_geometry_diagnostics()`.

- [ ] **Step 1: Write deterministic geometry tests**

```python
# tests/test_robot_bev_geometry.py
import numpy as np

from data_generation.robot_bev.geometry_checks import (
    history_to_current_lidar,
    points_to_bev_cells,
    project_lidar_to_image,
)


def test_opencv_projection_uses_z_forward():
    points = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]], dtype=np.float32)
    intrinsic = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]], dtype=np.float32)
    uv, valid = project_lidar_to_image(
        points, np.eye(4, dtype=np.float32), intrinsic, (80, 100)
    )
    np.testing.assert_allclose(uv[0], [50, 40])
    assert valid.tolist() == [True, False]


def test_base_forward_left_maps_to_row_column():
    points = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    rows, cols, valid = points_to_bev_cells(points)
    assert (rows[0], cols[0], valid[0]) == (50, 75, True)
    assert (rows[1], cols[1], valid[1]) == (100, 125, True)


def test_history_transform_matches_contract_formula():
    current_pose = np.eye(4, dtype=np.float32)
    current_pose[0, 3] = 1.0
    history_pose = np.eye(4, dtype=np.float32)
    transform = history_to_current_lidar(
        current_pose,
        np.eye(4, dtype=np.float32),
        history_pose,
        np.eye(4, dtype=np.float32),
    )
    np.testing.assert_allclose(transform[:3, 3], [-1.0, 0.0, 0.0])
```

- [ ] **Step 2: Run tests and verify missing module**

Run: `pytest -q tests/test_robot_bev_geometry.py`

Expected: FAIL with `No module named 'data_generation.robot_bev.geometry_checks'`.

- [ ] **Step 3: Implement exact geometry formulas**

```python
def history_to_current_lidar(cur_map_from_base, cur_base_from_lidar,
                             hist_map_from_base, hist_base_from_lidar):
    cur_map_from_lidar = cur_map_from_base @ cur_base_from_lidar
    hist_map_from_lidar = hist_map_from_base @ hist_base_from_lidar
    return np.linalg.inv(cur_map_from_lidar) @ hist_map_from_lidar


def points_to_bev_cells(points):
    rows = np.floor((points[:, 0] - 0.0) / 0.02).astype(np.int64)
    cols = np.floor((points[:, 1] + 1.5) / 0.02).astype(np.int64)
    valid = (rows >= 0) & (rows < 150) & (cols >= 0) & (cols < 150)
    return rows, cols, valid


def project_lidar_to_image(points_lidar, camera_from_lidar, intrinsic, image_shape):
    homogeneous = np.concatenate(
        [points_lidar[:, :3], np.ones((len(points_lidar), 1), dtype=np.float32)], axis=1
    )
    camera = (camera_from_lidar @ homogeneous.T).T[:, :3]
    pixels = (intrinsic @ camera.T).T
    uv = pixels[:, :2] / np.maximum(pixels[:, 2:3], 1e-8)
    height, width = image_shape
    valid = (
        (camera[:, 2] > 0)
        & (uv[:, 0] >= 0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0)
        & (uv[:, 1] < height)
    )
    return uv, valid
```

Implement `write_geometry_diagnostics(root, scene_id, frame_id, history_count=5)` to write an RGB point overlay, a BEV point/label/observed overlay, and an aligned-sweeps BEV image below `<root>/diagnostics/<scene_id>/`. Use fixed colors and include axis arrows labelled `x forward` and `y left`.

- [ ] **Step 4: Run geometry tests**

Run: `pytest -q tests/test_robot_bev_geometry.py`

Expected: `3 passed`.

- [ ] **Step 5: Commit geometry diagnostics**

```bash
git add data_generation/robot_bev/geometry_checks.py tests/test_robot_bev_geometry.py
git commit -m "feat: add robot BEV geometry diagnostics"
```

---

### Task 5: Refactor the Habitat-Sim Replica generator into source adapters

**Files:**
- Create: `data_generation/robot_bev/sources/__init__.py`
- Create: `data_generation/robot_bev/sources/habitat_common.py`
- Create: `data_generation/robot_bev/sources/replica.py`
- Create: `data_generation/robot_bev/cli/__init__.py`
- Create: `data_generation/robot_bev/cli/generate_replica.py`
- Test: `tests/test_robot_bev_replica_source.py`
- Reference: `data/generate_mydata/robot_bev_closed_loop.py`

**Interfaces:**
- Consumes: the existing validated Habitat-Sim 0.2.2 rendering behavior and `RobotBEVWriter`.
- Produces: `validate_replica_scene()`, `generate_scene()`, `run_generation()`, and `python -m data_generation.robot_bev.cli.generate_replica`.

- [ ] **Step 1: Write source-boundary tests**

```python
# tests/test_robot_bev_replica_source.py
import ast
from pathlib import Path

from data_generation.robot_bev.sources.replica import (
    MAP_CLASSES,
    semantic_category_to_map_class,
)


def test_replica_mapping_uses_canonical_classes():
    assert MAP_CLASSES == (
        "floor", "carpet", "obstacle", "wall", "furniture", "other"
    )
    assert semantic_category_to_map_class("table") == "furniture"
    assert semantic_category_to_map_class("wall") == "wall"
    assert semantic_category_to_map_class("rug") == "carpet"


def test_schema_writer_and_validator_do_not_import_habitat():
    for relative in ("schema.py", "writer.py", "validator.py", "geometry_checks.py"):
        path = Path("data_generation/robot_bev") / relative
        tree = ast.parse(path.read_text())
        imported = {
            node.names[0].name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import) and node.names
        }
        assert "habitat_sim" not in imported
```

- [ ] **Step 2: Run source tests and verify missing modules**

Run: `pytest -q tests/test_robot_bev_replica_source.py`

Expected: FAIL because `data_generation.robot_bev.sources.replica` does not exist.

- [ ] **Step 3: Move Habitat-common functions without behavior changes**

Move these functions and their required constants verbatim from the old script into `sources/habitat_common.py`:

```text
make_camera_intrinsic
rotation_x
camera_to_base_matrix
camera_optical_to_base_matrix
quat_to_rotation_matrix
map_from_base_matrix
map_from_habitat_pose
sensor_to_base_matrix
base_grid_to_habitat_local
transform_habitat_local_to_world
make_cfg
configure_navmesh_settings
initialize_navmesh
is_floor_level_safe
sample_safe_navigable_point
initialize_agent
depth_to_points
make_observation_mask
next_action
turn_agent_away
state_from_manifest
```

Keep Habitat imports inside this module. Replace local class constants with imports from `schema.py`. Do not move any pickle/index/sweep-writing function.

Guard the Habitat import so schema/mapping tests and CLI help can run without the rendering environment:

```python
try:
    import habitat_sim
except ImportError:
    habitat_sim = None


def require_habitat_sim():
    if habitat_sim is None:
        raise RuntimeError(
            "Habitat-Sim is required for rendering; activate the habitat022 environment"
        )
    return habitat_sim
```

Call `require_habitat_sim()` at the start of preflight/rendering functions. Avoid runtime-evaluated annotations such as `habitat_sim.Simulator` when Habitat is absent; quote those annotations or remove only the annotation while preserving behavior.

- [ ] **Step 4: Move Replica-specific functions and inject the writer**

Move these functions/classes into `sources/replica.py`:

```text
ReplicaSceneFiles
NavmeshTopdown
validate_replica_scene
load_scene_splits
build_semantic_id_to_class
semantic_category_to_map_class
build_navmesh_topdown
sample_navmesh_topdown
make_bev_labels
make_parser
```

Move old `generate()` as `generate_scene(args, scene_files, scene_split, writer)` and old `generate_all()` as `run_generation(args)`. The renamed functions are the only orchestration entry points used by the CLI.

Replace the old frame-output block with this writer boundary:

```python
payload = FramePayload(
    frame_id=frame_idx,
    timestamp=timestamp,
    rgb=rgb,
    points=points.astype(np.float32, copy=False),
    bev_labels=mask.astype(np.uint8, copy=False),
    observed_mask=valid_mask.astype(np.uint8, copy=False),
    class_validity=np.ones((len(MAP_CLASSES),), dtype=np.uint8),
    cam_intrinsic=intrinsic.astype(np.float32, copy=False),
    camera2base=camera_optical_to_base_matrix(t_base_camera_habitat),
    lidar2base=np.eye(4, dtype=np.float32),
    map_from_base=map_from_base_matrix(state),
    depth_mm=depth_mm,
    semantics=semantic_obs.astype(np.uint16, copy=False),
)
writer.write_frame(args.scene, scene_split, payload)
```

At scene completion call `writer.finalize_scene(args.scene, scene_split)`. After all scenes call `writer.finalize_dataset()`. Delete old `build_infos()`, `save_info_pickles()`, sweep construction, and absolute-path metadata from the new adapter. Keep rendering, navigation, semantic mapping, and label generation behavior unchanged.

Create one root writer before the scene loop, converting the scene-to-split assignment into the writer contract:

```python
split_lists = {
    split: [scene for scene in scenes if assignments[scene] == split]
    for split in ("train", "val", "test")
}
writer = RobotBEVWriter(
    root=Path(args.output_dir),
    dataset_id=args.dataset_id,
    source_type="simulation",
    source_dataset="replica_v1",
    generator_name="habitat_replica_robot_bev",
    generator_version="3",
    splits=split_lists,
    generation_parameters=generation_parameters(args),
    resume=args.resume,
)
```

Implement `generation_parameters(args)` as a plain dictionary containing dataset config path, requested scenes, split-file contents, image width/height/HFOV, camera/agent/navmesh values, BEV bounds, obstacle thresholds, depth/point settings, motion values, seed/timestamps, physics mode, and Habitat-Sim version. Exclude only `output_dir`, `resume`, `preflight_only`, `gpu_id`, and visualization-output flags because they do not change training artifacts.

- [ ] **Step 5: Add the Replica CLI and remove generation-stage sweeps**

The CLI parser retains every existing rendering/navmesh/trajectory option except `--num-sweeps`. It adds required `--dataset-id` and passes all generation-affecting parameters to the writer fingerprint.

```python
# data_generation/robot_bev/cli/generate_replica.py
from data_generation.robot_bev.sources.replica import make_parser, run_generation


def main() -> None:
    args = make_parser().parse_args()
    run_generation(args)


if __name__ == "__main__":
    main()
```

Run: `python -m data_generation.robot_bev.cli.generate_replica --help`

Expected: exit 0; help includes `--dataset-id`, `--split-file`, `--resume`, and `--preflight-only`, and does not include `--num-sweeps`.

- [ ] **Step 6: Run dependency-light and Habitat preflight tests**

Run in the normal Python 3.8 test environment:

`pytest -q tests/test_robot_bev_schema.py tests/test_robot_bev_writer.py tests/test_robot_bev_validator.py tests/test_robot_bev_geometry.py tests/test_robot_bev_replica_source.py`

Expected: all tests PASS.

Run in `habitat022`:

```bash
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset /path/to/replica.scene_dataset_config.json \
  --dataset-id replica_robot_bev_v3 \
  --scene office_1 \
  --output-dir /tmp/replica_robot_bev_v3_preflight \
  --preflight-only
```

Expected: exit 0 and log `Preflight OK: office_1` with Habitat-Sim version 0.2.2 and PTex atlas count.

- [ ] **Step 7: Commit the refactored source adapter**

```bash
git add data_generation/robot_bev/sources data_generation/robot_bev/cli \
  tests/test_robot_bev_replica_source.py
git commit -m "refactor: split Replica robot BEV generator"
```

---

### Task 6: Validation CLI, tracked examples, documentation, and 10-frame smoke output

**Files:**
- Create: `data_generation/robot_bev/cli/validate_dataset.py`
- Create: `data_generation/robot_bev/configs/replica_scenes.txt`
- Create: `data_generation/robot_bev/configs/replica_splits.example.json`
- Create: `data_generation/robot_bev/README.md`
- Create: `data_generation/robot_bev/docs/schema_v3.md`
- Create: `data_generation/robot_bev/docs/habitat_replica.md`
- Create: `data_generation/robot_bev/docs/add_new_source.md`
- Create: `data_generation/robot_bev/docs/quality_checks.md`
- Test: `tests/test_robot_bev_cli.py`

**Interfaces:**
- Consumes: canonical dataset roots and optional geometry sample selection.
- Produces: a JSON validation report, diagnostic images, exact smoke/full-generation commands, and a source-adapter checklist.

- [ ] **Step 1: Write CLI tests**

```python
# tests/test_robot_bev_cli.py
import subprocess
import sys


def test_validation_cli_help():
    completed = subprocess.run(
        [sys.executable, "-m", "data_generation.robot_bev.cli.validate_dataset", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "--root" in completed.stdout
    assert "--geometry-scene" in completed.stdout
```

- [ ] **Step 2: Run CLI test and verify missing module**

Run: `pytest -q tests/test_robot_bev_cli.py`

Expected: FAIL because the validation CLI module does not exist.

- [ ] **Step 3: Implement validation CLI**

```python
import argparse
import json

from data_generation.robot_bev.geometry_checks import write_geometry_diagnostics
from data_generation.robot_bev.validator import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a canonical Robot BEV dataset")
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--geometry-scene")
    parser.add_argument("--geometry-frame", type=int, default=0)
    args = parser.parse_args()
    report = validate_dataset(args.root, args.split)
    print(json.dumps({
        "valid": report.valid,
        "frame_counts": report.frame_counts,
        "warnings": report.warnings,
    }, indent=2))
    if args.geometry_scene:
        write_geometry_diagnostics(
            args.root, args.geometry_scene, args.geometry_frame, history_count=5
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add exact tracked Replica split examples**

Copy the 18 scene IDs and 14/2/2 split approved in the design into `configs/replica_scenes.txt` and `configs/replica_splits.example.json`. Assert with a test that all 18 scene names occur once and that counts are 14 train, 2 val, 2 test.

- [ ] **Step 5: Write the four operational documents**

Document these exact commands, replacing only absolute dataset/output paths at runtime:

```bash
conda activate habitat022
python -m data_generation.robot_bev.cli.generate_replica \
  --dataset /path/to/replica.scene_dataset_config.json \
  --dataset-id replica_robot_bev_v3_smoke \
  --scene office_1 \
  --split-file data_generation/robot_bev/configs/replica_splits.example.json \
  --output-dir /data/replica_robot_bev_v3_smoke \
  --num-frames 10 \
  --gpu-id 0 \
  --disable-physics \
  --recompute-navmesh \
  --save-visualization

python -m data_generation.robot_bev.cli.validate_dataset \
  --root /data/replica_robot_bev_v3_smoke \
  --geometry-scene office_1 \
  --geometry-frame 0
```

Also document the 18×600 command, safe `--resume` rules, fixed coordinate/schema tables, the adapter method boundary, required semantic mapping behavior, fatal-vs-warning quality policy, and the requirement to inspect one geometry bundle for every new source adapter.

- [ ] **Step 6: Run the complete toolkit suite**

Run: `pytest -q tests/test_robot_bev_schema.py tests/test_robot_bev_writer.py tests/test_robot_bev_validator.py tests/test_robot_bev_geometry.py tests/test_robot_bev_replica_source.py tests/test_robot_bev_cli.py`

Expected: all tests PASS.

Run `git diff --check` and expect no output.

- [ ] **Step 7: Generate and validate the 10-frame smoke dataset in `habitat022`**

Run the documented generation and validation commands.

Expected:

- root `multi_scene_summary.json` has status `complete` and 10 frames in its assigned split;
- validator JSON reports `"valid": true`;
- camera, BEV, and sweep diagnostic images exist;
- a human confirms x-forward/y-left orientation and camera projection.

- [ ] **Step 8: Commit toolkit docs and CLI**

```bash
git add data_generation/robot_bev/README.md data_generation/robot_bev/configs \
  data_generation/robot_bev/docs data_generation/robot_bev/cli/validate_dataset.py \
  tests/test_robot_bev_cli.py
git commit -m "docs: add robot BEV generation workflow"
```

- [ ] **Step 9: Retire the ignored copied generator and smoke data after the new gate passes**

First verify the exact directories:

```bash
du -sh data/generate_mydata data/replica_observed_v3
find data/generate_mydata -maxdepth 2 -type f -print
```

After the new generator is committed and the new 10-frame root validates successfully, remove only the explicitly superseded local copies:

```bash
rm -rf data/generate_mydata data/replica_observed_v3
```

The remote operator applies the same policy to the old 18×600 output only after copying the new code and confirming a 10-frame remote smoke run.

## Plan 1 Completion Gate

Before starting plan 2, require all dependency-light tests to pass, a Replica 10-frame canonical dataset to validate successfully, and a human to approve its camera/BEV/sweep diagnostics. Do not regenerate the remote 18×600 dataset until this gate passes.
