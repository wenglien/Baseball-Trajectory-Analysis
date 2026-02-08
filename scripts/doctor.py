"""
Environment doctor for speedgun-mobile.

This script helps diagnose common setup issues:
- Wrong Python version
- mediapipe shadowed by local files/folders
- Missing key dependencies
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def _try_import(name: str):
    try:
        return importlib.import_module(name), None
    except Exception as e:  # noqa: BLE001
        return None, e


def _print_kv(k: str, v: str) -> None:
    print(f"{k}: {v}")


def main() -> int:
    print("== speedgun-mobile doctor ==")
    _print_kv("python", sys.version.replace("\n", " "))
    _print_kv("executable", sys.executable)
    _print_kv("platform", f"{platform.system()} {platform.release()} ({platform.machine()})")
    _print_kv("cwd", str(Path.cwd()))
    print()

    repo_root = Path(__file__).resolve().parents[1]
    _print_kv("repo_root", str(repo_root))

    # Common shadowing checks
    shadow_candidates = [
        repo_root / "mediapipe.py",
        repo_root / "mediapipe",
        repo_root / "ultralytics.py",
        repo_root / "ultralytics",
    ]
    shadow_hits = [p for p in shadow_candidates if p.exists()]
    if shadow_hits:
        print("\nWARN 可能的同名覆蓋（會導致 import 載入錯誤）：")
        for p in shadow_hits:
            print(f" - {p}")

    print("\n== imports ==")
    for name in ["cv2", "mediapipe", "ultralytics", "numpy"]:
        mod, err = _try_import(name)
        if err is not None:
            print(f"FAIL import {name} 失敗：{err}")
            continue

        mod_file = getattr(mod, "__file__", None)
        mod_ver = getattr(mod, "__version__", None)
        print(f"OK {name}: version={mod_ver} file={mod_file}")

        if name == "mediapipe":
            has_solutions = hasattr(mod, "solutions")
            print(f"   mediapipe.has_solutions: {has_solutions}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

