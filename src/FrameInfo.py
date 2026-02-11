from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class FrameInfo:
    """Single frame data with optional ball detection info."""

    frame: np.ndarray
    ball_in_frame: bool
    ball: Tuple[int, int] = (0, 0)
    ball_color: Tuple[int, int, int] = (0, 0, 0)
    ball_lost_tracking: bool = False
