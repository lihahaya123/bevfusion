"""Validate canonical Robot BEV datasets and write optional diagnostics."""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

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
    parser.add_argument("--geometry-frame", type=int, default=0)
    return parser


def _success_payload(
    root: Path,
    split: Optional[str],
    geometry_scene: Optional[str],
    geometry_frame: int,
) -> Dict[str, object]:
    report = validate_dataset(root, split)
    payload: Dict[str, object] = {
        "valid": report.valid,
        "frame_counts": report.frame_counts,
        "warnings": report.warnings,
    }
    if geometry_scene:
        paths = write_geometry_diagnostics(
            root,
            geometry_scene,
            geometry_frame,
            history_count=5,
        )
        payload["geometry_diagnostics"] = [str(path) for path in paths]
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = make_parser().parse_args(argv)
        payload = _success_payload(
            args.root,
            args.split,
            args.geometry_scene,
            args.geometry_frame,
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
