"""Tests for the WildlifeDetector module using mock models and synthetic frames."""

from __future__ import annotations

import numpy as np
import pytest

from detector import DEFAULT_ANIMAL_CLASSES, WildlifeDetector, draw_detections
from tests.conftest import MockBoxData, MockResult, MockYOLOModel


def test_detector_lazy_loading() -> None:
    """Verify WildlifeDetector does not load YOLO on instantiation."""
    detector = WildlifeDetector(model=None, model_name="dummy_nonexistent.pt")
    # Underlying private model attribute should initially be None
    assert detector._model is None


def test_detector_uses_injected_stub(synthetic_frame: np.ndarray) -> None:
    """Verify detector immediately uses injected stub without loading YOLO."""
    stub = MockYOLOModel()
    detector = WildlifeDetector(model=stub)
    assert detector.model is stub

    detections = detector.detect_and_track(synthetic_frame)
    assert isinstance(detections, list)
    assert stub.call_count == 1


def test_detector_filters_non_animal_classes(
    synthetic_frame: np.ndarray,
    standard_names: dict[int, str],
) -> None:
    """Verify non-animal classes (person, car) are filtered out, keeping only animals."""
    boxes = MockBoxData(
        xyxy=[[10, 10, 50, 50], [60, 60, 100, 100], [110, 110, 150, 150], [160, 160, 200, 200]],
        conf=[0.9, 0.85, 0.95, 0.8],
        cls=[0, 2, 22, 14],  # person (0), car (2), zebra (22), bird (14)
        track_id=[1, 2, 3, 4],
    )
    result = MockResult(boxes=boxes, names=standard_names)
    stub = MockYOLOModel(results_sequence=[result])

    detector = WildlifeDetector(model=stub)
    detections = detector.detect_and_track(synthetic_frame)

    # Should only retain zebra and bird
    assert len(detections) == 2
    detected_species = {d["species"] for d in detections}
    assert detected_species == {"zebra", "bird"}

    # Validate retained detection data
    zebra_det = next(d for d in detections if d["species"] == "zebra")
    assert zebra_det["track_id"] == 3
    assert zebra_det["confidence"] == 0.95
    assert zebra_det["bbox"] == [110.0, 110.0, 150.0, 150.0]


def test_detector_custom_target_classes(
    synthetic_frame: np.ndarray,
    standard_names: dict[int, str],
) -> None:
    """Verify that specifying target_classes restricts detection to that subset."""
    boxes = MockBoxData(
        xyxy=[[10, 10, 50, 50], [60, 60, 100, 100]],
        conf=[0.9, 0.9],
        cls=[22, 14],  # zebra (22), bird (14)
        track_id=[1, 2],
    )
    result = MockResult(boxes=boxes, names=standard_names)
    stub = MockYOLOModel(results_sequence=[result])

    # Allow only zebra
    detector = WildlifeDetector(model=stub, target_classes=["zebra"])
    detections = detector.detect_and_track(synthetic_frame)

    assert len(detections) == 1
    assert detections[0]["species"] == "zebra"


def test_detector_confidence_threshold_filtering(
    synthetic_frame: np.ndarray,
    standard_names: dict[int, str],
) -> None:
    """Verify detections below confidence threshold are omitted."""
    boxes = MockBoxData(
        xyxy=[[10, 10, 50, 50], [60, 60, 100, 100]],
        conf=[0.20, 0.80],  # 0.20 should be dropped
        cls=[22, 22],
        track_id=[1, 2],
    )
    result = MockResult(boxes=boxes, names=standard_names)
    stub = MockYOLOModel(results_sequence=[result])

    detector = WildlifeDetector(model=stub, conf_threshold=0.50)
    detections = detector.detect_and_track(synthetic_frame)

    assert len(detections) == 1
    assert detections[0]["track_id"] == 2
    assert detections[0]["confidence"] == 0.80


def test_detector_output_dict_structure(
    synthetic_frame: np.ndarray,
    standard_names: dict[int, str],
) -> None:
    """Verify detection items contain all required keys with correct types."""
    boxes = MockBoxData(
        xyxy=[[5.5, 10.2, 50.8, 80.1]],
        conf=[0.8765],
        cls=[20],  # elephant
        track_id=[42],
    )
    result = MockResult(boxes=boxes, names=standard_names)
    stub = MockYOLOModel(results_sequence=[result])

    detector = WildlifeDetector(model=stub)
    detections = detector.detect_and_track(synthetic_frame)

    assert len(detections) == 1
    det = detections[0]
    assert set(det.keys()) == {"track_id", "species", "confidence", "bbox"}
    assert isinstance(det["track_id"], int)
    assert det["track_id"] == 42
    assert isinstance(det["species"], str)
    assert det["species"] == "elephant"
    assert isinstance(det["confidence"], float)
    assert det["confidence"] == 0.8765
    assert isinstance(det["bbox"], list)
    assert len(det["bbox"]) == 4


def test_detector_empty_detections(synthetic_frame: np.ndarray) -> None:
    """Verify empty boxes return an empty detection list."""
    empty_boxes = MockBoxData(xyxy=[], conf=[], cls=[], track_id=[])
    stub = MockYOLOModel(results_sequence=[MockResult(boxes=empty_boxes)])

    detector = WildlifeDetector(model=stub)
    detections = detector.detect_and_track(synthetic_frame)
    assert detections == []


def test_draw_detections_renders_on_frame(synthetic_frame: np.ndarray) -> None:
    """Verify draw_detections modifies pixels and returns valid annotated array."""
    detections = [
        {"track_id": 1, "species": "zebra", "confidence": 0.95, "bbox": [50, 50, 150, 150]},
        {"track_id": 2, "species": "bird", "confidence": 0.80, "bbox": [200, 200, 250, 250]},
    ]
    annotated = draw_detections(synthetic_frame, detections)

    assert annotated.shape == synthetic_frame.shape
    assert annotated.dtype == synthetic_frame.dtype
    # The output frame should no longer be completely black
    assert np.any(annotated > 0)
