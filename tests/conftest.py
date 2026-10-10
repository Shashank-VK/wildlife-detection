"""Pytest fixtures and mock classes for Wildlife Detection tests.

Ensures all tests execute deterministically and offline with zero model downloads.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pytest

from detector import WildlifeDetector


class MockBoxData:
    """Mock container mimicking Ultralytics Boxes."""

    def __init__(
        self,
        xyxy: Sequence[Sequence[float]] | np.ndarray,
        conf: Sequence[float] | np.ndarray,
        cls: Sequence[int] | np.ndarray,
        track_id: Sequence[int | None] | np.ndarray | None = None,
    ) -> None:
        self.xyxy = np.array(xyxy, dtype=float) if len(xyxy) > 0 else np.empty((0, 4), dtype=float)
        self.conf = np.array(conf, dtype=float) if len(conf) > 0 else np.empty((0,), dtype=float)
        self.cls = np.array(cls, dtype=int) if len(cls) > 0 else np.empty((0,), dtype=int)
        if track_id is not None and len(track_id) > 0:
            self.id = np.array(track_id, dtype=float)
        else:
            self.id = None

    def __len__(self) -> int:
        return len(self.cls)


class MockResult:
    """Mock container mimicking Ultralytics Results."""

    def __init__(self, boxes: MockBoxData, names: dict[int, str] | None = None) -> None:
        self.boxes = boxes
        self.names = names or {
            0: "person",
            1: "bicycle",
            2: "car",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
        }


class MockYOLOModel:
    """Mock YOLO model returning scripted Results sequences."""

    def __init__(
        self,
        results_sequence: Sequence[MockResult] | None = None,
        names: dict[int, str] | None = None,
    ) -> None:
        self.names = names or {
            0: "person",
            1: "bicycle",
            2: "car",
            14: "bird",
            15: "cat",
            16: "dog",
            17: "horse",
            18: "sheep",
            19: "cow",
            20: "elephant",
            21: "bear",
            22: "zebra",
            23: "giraffe",
        }
        self.results_sequence = list(results_sequence) if results_sequence is not None else []
        self.call_count = 0

    def track(self, source: Any, **kwargs: Any) -> list[MockResult]:
        """Mock track() call matching Ultralytics YOLO.track API."""
        if self.call_count < len(self.results_sequence):
            res = self.results_sequence[self.call_count]
        elif self.results_sequence:
            res = self.results_sequence[-1]
        else:
            res = MockResult(MockBoxData([], [], []), self.names)

        self.call_count += 1
        return [res]


@pytest.fixture
def synthetic_frame() -> np.ndarray:
    """Create a synthetic 480x640 BGR image array."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def standard_names() -> dict[int, str]:
    """Standard COCO class mapping for mock results."""
    return {
        0: "person",
        1: "bicycle",
        2: "car",
        14: "bird",
        15: "cat",
        16: "dog",
        17: "horse",
        18: "sheep",
        19: "cow",
        20: "elephant",
        21: "bear",
        22: "zebra",
        23: "giraffe",
    }
