"""
Quick smoke test (no heavy runtime) for speedgun-mobile.

Goals:
- Verify key modules can be imported
- Print versions/paths for common dependency debugging
- Avoid opening GUI windows or running long inference
"""

from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path


def _try_import(name: str):
    try:
        mod = importlib.import_module(name)
        return True, mod, None
    except Exception as e:  # noqa: BLE001
        return False, None, e


def main() -> int:
    # Ensure repo root on sys.path when executed as `python scripts/smoke_test.py`
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("== speedgun-mobile smoke test ==")
    print(f"python: {sys.version.replace(chr(10), ' ')}")
    print(f"executable: {sys.executable}")
    print(f"platform: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"cwd: {Path.cwd()}")
    print(f"repo_root: {repo_root}")
    print()

    required = ["numpy", "cv2", "mediapipe", "ultralytics"]
    ok_all = True

    print("== dependency imports ==")
    for name in required:
        ok, mod, err = _try_import(name)
        if not ok:
            ok_all = False
            print(f"{name}: import 失敗：{err}")
            continue
        ver = getattr(mod, "__version__", None)
        path = getattr(mod, "__file__", None)
        print(f" {name}: version={ver} file={path}")
        if name == "mediapipe":
            print(f"   mediapipe.has_solutions: {hasattr(mod, 'solutions')}")

    print("\n== project imports ==")
    project_modules = [
        "src.get_pitch_frames_yolov8",
        "src.pipelines.yolov8_pipeline",
        "pitching_overlay",
    ]
    for name in project_modules:
        ok, _mod, err = _try_import(name)
        if not ok:
            ok_all = False
            print(f"{name}: import 失敗：{err}")
        else:
            print(f" {name}")

    print("\n== optional (legacy) ==")
    ok_tf, tf_mod, tf_err = _try_import("tensorflow")
    if not ok_tf:
        print("INFO tensorflow: 未安裝（只有使用 YOLOv4 legacy / CoreML 轉換才需要）")
    else:
        print(
            f"tensorflow: version={getattr(tf_mod, '__version__', None)} file={getattr(tf_mod, '__file__', None)}"
        )

    print()
    if ok_all:
        print("Smoke test PASS")
        return 0
    print("Smoke test FAIL（請先修正上述 import 問題）")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

