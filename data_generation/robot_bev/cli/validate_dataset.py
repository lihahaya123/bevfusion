"""Validate canonical Robot BEV datasets and write optional diagnostics."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from data_generation.robot_bev.geometry_checks import write_geometry_diagnostics
from data_generation.robot_bev.validator import (
    DatasetValidationError,
    validate_dataset,
)


class ArgumentParseError(ValueError):
    """Raised when validation CLI arguments cannot be parsed."""


class _ValidationArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ArgumentParseError(message)


def make_parser() -> argparse.ArgumentParser:
    parser = _ValidationArgumentParser(
        description="Validate a canonical Robot BEV dataset"
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--geometry-scene")
    parser.add_argument(
        "--geometry-all-scenes",
        action="store_true",
        help=(
            "Write geometry diagnostics for every scene in the selected split. "
            "If --split is omitted, all scenes from all splits are used."
        ),
    )
    parser.add_argument("--geometry-frame", type=int, default=0)
    parser.add_argument(
        "--geometry-frame-range",
        type=int,
        nargs=3,
        metavar=("START", "STOP", "STEP"),
        help=(
            "Write geometry diagnostics for range(START, STOP, STEP). "
            "Requires --geometry-scene or --geometry-all-scenes."
        ),
    )
    return parser


def _load_geometry_scenes(root: Path, split: Optional[str]) -> List[str]:
    splits_path = Path(root).expanduser().resolve() / "splits.json"
    try:
        splits = json.loads(splits_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read dataset splits JSON: {splits_path}") from exc

    selected = (split,) if split is not None else ("train", "val", "test")
    scenes: List[str] = []
    seen = set()
    for split_name in selected:
        values = splits.get(split_name, [])
        if not isinstance(values, list) or not all(
            isinstance(scene, str) for scene in values
        ):
            raise ValueError(f"splits.json field {split_name!r} must be a list")
        for scene in values:
            if scene not in seen:
                scenes.append(scene)
                seen.add(scene)
    return scenes


def _geometry_frames(
    geometry_scene: Optional[str],
    geometry_all_scenes: bool,
    geometry_frame: int,
    geometry_frame_range: Optional[Sequence[int]],
) -> List[int]:
    if geometry_scene and geometry_all_scenes:
        raise ValueError(
            "--geometry-scene and --geometry-all-scenes are mutually exclusive"
        )
    if not geometry_scene and not geometry_all_scenes:
        if geometry_frame_range is not None:
            raise ValueError(
                "--geometry-frame-range requires --geometry-scene or "
                "--geometry-all-scenes"
            )
        return []
    if geometry_frame_range is None:
        return [geometry_frame]
    start, stop, step = geometry_frame_range
    if step == 0:
        raise ValueError("--geometry-frame-range STEP must not be zero")
    frames = list(range(start, stop, step))
    if not frames:
        raise ValueError(
            "--geometry-frame-range produced no frames; check START/STOP/STEP"
        )
    return frames


def _success_payload(
    root: Path,
    split: Optional[str],
    geometry_scene: Optional[str],
    geometry_all_scenes: bool,
    geometry_frame: int,
    geometry_frame_range: Optional[Sequence[int]],
) -> Dict[str, object]:
    report = validate_dataset(root, split)
    payload: Dict[str, object] = {
        "valid": report.valid,
        "frame_counts": report.frame_counts,
        "warnings": report.warnings,
    }
    frames = _geometry_frames(
        geometry_scene,
        geometry_all_scenes,
        geometry_frame,
        geometry_frame_range,
    )
    if not frames:
        return payload

    scenes = (
        _load_geometry_scenes(root, split)
        if geometry_all_scenes
        else [str(geometry_scene)]
    )
    if not scenes:
        raise ValueError("No scenes selected for geometry diagnostics")

    if len(scenes) == 1 and len(frames) == 1 and not geometry_frame_range:
        paths = write_geometry_diagnostics(
            root, scenes[0], frames[0], history_count=5
        )
        payload["geometry_diagnostics"] = [str(path) for path in paths]
        return payload

    if len(scenes) == 1 and not geometry_all_scenes:
        diagnostics = {}
        for frame in frames:
            paths = write_geometry_diagnostics(
                root, scenes[0], frame, history_count=5
            )
            diagnostics[str(frame)] = [str(path) for path in paths]
        payload["geometry_diagnostics"] = diagnostics
        return payload

    diagnostics = {}
    for scene in scenes:
        scene_diagnostics = {}
        for frame in frames:
            paths = write_geometry_diagnostics(
                root, scene, frame, history_count=5
            )
            scene_diagnostics[str(frame)] = [str(path) for path in paths]
        diagnostics[scene] = scene_diagnostics
    payload["geometry_diagnostics"] = diagnostics
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = make_parser().parse_args(argv)
        payload = _success_payload(
            args.root,
            args.split,
            args.geometry_scene,
            args.geometry_all_scenes,
            args.geometry_frame,
            args.geometry_frame_range,
        )
    except (ArgumentParseError, DatasetValidationError, OSError, ValueError) as error:
        json.dump(
            {
                "valid": False,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            sys.stderr,
            indent=2,
        )
        sys.stderr.write("\n")
        return 1
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
