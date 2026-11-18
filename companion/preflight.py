"""Inspect local observation prerequisites without loading models or opening devices."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .runtime import default_weights, file_digest

DEPENDENCIES = ("cv2", "numpy", "torch", "ultralytics", "serial")


def inspect_environment(weights: Path, discover=None) -> dict:
    discover = discover or importlib.util.find_spec
    dependencies = {}
    for name in DEPENDENCIES:
        try:
            dependencies[name] = {"discoverable": discover(name) is not None}
        except (ImportError, ValueError, OSError) as error:
            dependencies[name] = {"discoverable": False, "error_type": type(error).__name__}
    model = {"path": str(weights.resolve()), "file_available": False, "sha256": None}
    try:
        if weights.suffix.lower() != ".pt":
            model["problem"] = "expected_local_pt_file"
        elif not weights.is_file() or weights.stat().st_size == 0:
            model["problem"] = "missing_or_empty_file"
        else:
            model.update(file_available=True, sha256=file_digest(weights))
    except OSError as error:
        model["problem"] = type(error).__name__
    missing = [name for name in DEPENDENCIES if not dependencies[name]["discoverable"]]
    # Discovery does not prove binary imports, model labels, or live device operation.
    observation_missing = [name for name in missing if name != "serial"]
    return {
        "schema_version": 1,
        "python": {"executable": sys.executable, "version": sys.version.split()[0],
                   "supported": sys.version_info >= (3, 10)},
        "dependencies": dependencies,
        "missing_observation_dependencies": observation_missing,
        "optional_serial_dependency_missing": "serial" in missing,
        "model": model,
        "local_prerequisites_found": (sys.version_info >= (3, 10)
                                      and not observation_missing and model["file_available"]),
        "checks_not_performed": ["binary_imports", "model_deserialization_and_class_validation",
                                 "camera_or_network_access", "serial_open", "firmware_build",
                                 "voltage_and_harness_measurements", "actuator_tests"],
        "hardware_validated": False,
        "actuator_control_enabled": False,
        "power_on_approved": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=default_weights())
    parser.add_argument("--output", type=Path, help="optional new JSON report; never overwritten")
    args = parser.parse_args(argv)
    report = inspect_environment(args.weights)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        except OSError as error:
            parser.error(f"cannot create new report: {error}")
    print(rendered, end="")
    return 0 if report["local_prerequisites_found"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
