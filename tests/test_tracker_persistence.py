"""Integration tests verifying end-to-end multi-frame tracker persistence and pipeline execution."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from counter import LiveCounter
from detector import WildlifeDetector
from logger import DetectionLogger
from main import run_pipeline
from tests.conftest import MockBoxData, MockResult, MockYOLOModel


def test_tracker_id_persistence_multi_frame(standard_names: dict[int, str]) -> None:
    """Verify persistent tracker IDs across a 5-frame sequence maintain count consistency."""
    # Frame 1: Zebra #1 enters
    f1_boxes = MockBoxData(xyxy=[[50, 50, 100, 100]], conf=[0.9], cls=[22], track_id=[1])
    # Frame 2: Zebra #1 continues, Elephant #2 enters
    f2_boxes = MockBoxData(
        xyxy=[[55, 52, 105, 102], [200, 200, 300, 300]],
        conf=[0.92, 0.88],
        cls=[22, 20],
        track_id=[1, 2],
    )
    # Frame 3: Both continue
    f3_boxes = MockBoxData(
        xyxy=[[60, 55, 110, 105], [205, 202, 305, 302]],
        conf=[0.91, 0.89],
        cls=[22, 20],
        track_id=[1, 2],
    )
    # Frame 4: Zebra #1 leaves, Elephant #2 continues
    f4_boxes = MockBoxData(xyxy=[[210, 205, 310, 305]], conf=[0.87], cls=[20], track_id=[2])
    # Frame 5: Elephant #2 continues
    f5_boxes = MockBoxData(xyxy=[[215, 208, 315, 308]], conf=[0.86], cls=[20], track_id=[2])

    results = [
        MockResult(boxes=b, names=standard_names)
        for b in [f1_boxes, f2_boxes, f3_boxes, f4_boxes, f5_boxes]
    ]
    mock_model = MockYOLOModel(results_sequence=results)
    detector = WildlifeDetector(model=mock_model)
    counter = LiveCounter()
    logger = DetectionLogger()

    # Process 5 synthetic frames
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(1, 6):
        dets = detector.detect_and_track(frame, persist=True)
        counter.update(dets, frame_no=i, timestamp=f"2026-10-10T12:00:0{i}Z")
        logger.log(frame_no=i, detections=dets, timestamp=f"2026-10-10T12:00:0{i}Z")

    final_snap = counter.get_snapshot()

    # Total unique seen must be exactly 2 (zebra #1 and elephant #2)
    assert final_snap["total_unique"] == 2
    assert final_snap["cumulative_counts"] == {"zebra": 1, "elephant": 1}

    # Current in-frame at frame 5 should be 1 (only elephant #2)
    assert final_snap["total_current"] == 1
    assert final_snap["current_counts"] == {"elephant": 1}

    # Verify event sequences
    events = counter.get_events()
    # Expect: entered #1 (frame 1), entered #2 (frame 2), left #1 (frame 4)
    entered_events = [e for e in events if e["event"] == "entered"]
    left_events = [e for e in events if e["event"] == "left"]

    assert len(entered_events) == 2
    assert {e["track_id"] for e in entered_events} == {1, 2}

    assert len(left_events) == 1
    assert left_events[0]["track_id"] == 1
    assert left_events[0]["frame_no"] == 4

    # Logger records should total 1 + 2 + 2 + 1 + 1 = 7 detections
    assert len(logger) == 7


def test_cli_pipeline_with_synthetic_video(tmp_path: Path, standard_names: dict[int, str]) -> None:
    """Verify run_pipeline executes end-to-end on a synthetic video file without external network calls."""
    # Create a synthetic AVI video file (using 'MJPG' codec supported everywhere)
    video_path = tmp_path / "synthetic_test_video.avi"
    csv_path = tmp_path / "output_detections.csv"

    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (width, height))

    num_frames = 6
    for i in range(num_frames):
        synth_frame = np.full((height, width, 3), i * 30, dtype=np.uint8)
        writer.write(synth_frame)
    writer.release()

    # Set up mock results for the 6 frames
    mock_boxes = MockBoxData(
        xyxy=[[20, 20, 80, 80]],
        conf=[0.93],
        cls=[22],  # zebra
        track_id=[10],
    )
    mock_model = MockYOLOModel(
        results_sequence=[MockResult(boxes=mock_boxes, names=standard_names)] * num_frames
    )
    detector = WildlifeDetector(model=mock_model)

    # Run the pipeline
    final_counter, final_logger = run_pipeline(
        source=str(video_path),
        output_path=csv_path,
        detector=detector,
        show=False,
        max_frames=num_frames,
    )

    # Verify outputs
    assert csv_path.exists()
    snap = final_counter.get_snapshot()
    assert snap["total_unique"] == 1
    assert snap["cumulative_counts"]["zebra"] == 1
    assert len(final_logger) == num_frames
