from __future__ import annotations

import logging
from typing import Optional


def configure_logging(*, verbose: int = 0, quiet: bool = False) -> None:
    """
    設定全域 logging。

    - quiet: 只顯示 ERROR
    - verbose: 0=INFO, 1+=DEBUG
    """
    if quiet:
        level = logging.ERROR
    else:
        level = logging.DEBUG if verbose and verbose > 0 else logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    # 避免重複加 handler（例如在同一個 Python process 中多次呼叫）
    if root.handlers:
        for h in root.handlers:
            h.setLevel(level)
        return

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name if name else "speedgun")

