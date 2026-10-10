"""Tests for DetectionLogger: DataFrame logging, CSV export, species distributions, and peak activity."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from logger import DetectionLogger


def test_logger_dataframe_shape_and_columns() -> None:
    """Verify logger records detections and exposes expected DataFrame structure."""
    logger = DetectionLogger()
    detections_frame_1 = [
        {"track_id": 1, "species": "zebra", "confidence": 0.95, "bbox": [10.0, 20.0, 100.0, 200.0]},
        {"track_id": 2, "species": "bird", "confidence": 0.82, "bbox": [30.0, 40.0, 50.0, 60.0]},
    ]
    detections_frame_2 = [
        {"track_id": 1, "species": "zebra", "confidence": 0.94, "bbox": [12.0, 22.0, 102.0, 202.0]},
    ]

    logger.log(frame_no=1, detections=detections_frame_1, timestamp="2026-10-10T12:00:00Z")
    logger.log(frame_no=2, detections=detections_frame_2, timestamp="2026-10-10T12:00:01Z")

    df = logger.dataframe
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert set(df.columns) == {"timestamp", "frame_no", "track_id", "species", "confidence", "bbox"}
    assert list(df["frame_no"]) == [1, 1, 2]
    assert list(df["species"]) == ["zebra", "bird", "zebra"]


def test_logger_csv_round_trip(tmp_path: Path) -> None:
    """Verify to_csv writes valid CSV and loads back matching data."""
    csv_file = tmp_path / "test_detections.csv"
    logger = DetectionLogger()
    detections = [
        {"track_id": 10, "species": "elephant", "confidence": 0.91, "bbox": [50.0, 60.0, 150.0, 160.0]},
        {"track_id": 11, "species": "giraffe", "confidence": 0.88, "bbox": [200.0, 210.0, 300.0, 310.0]},
    ]
    logger.log(frame_no=5, detections=detections, timestamp="2026-10-10T12:30:00Z")

    logger.to_csv(csv_file)
    assert csv_file.exists()

    # Read back via pandas directly
    loaded_df = pd.read_csv(csv_file)
    assert len(loaded_df) == 2
    assert set(loaded_df["species"]) == {"elephant", "giraffe"}

    # Read back via DetectionLogger.from_csv
    restored_logger = DetectionLogger.from_csv(csv_file)
    assert len(restored_logger) == 2
    assert list(restored_logger.dataframe["species"]) == ["elephant", "giraffe"]


def test_logger_species_distribution() -> None:
    """Verify species_distribution returns sorted DataFrame of counts per species."""
    logger = DetectionLogger()
    detections = [
        {"track_id": 1, "species": "zebra", "confidence": 0.9},
        {"track_id": 2, "species": "zebra", "confidence": 0.85},
        {"track_id": 3, "species": "zebra", "confidence": 0.92},
        {"track_id": 4, "species": "bird", "confidence": 0.80},
        {"track_id": 5, "species": "elephant", "confidence": 0.99},
        {"track_id": 6, "species": "elephant", "confidence": 0.95},
    ]
    logger.log(frame_no=1, detections=detections, timestamp="2026-10-10T12:00:00Z")

    dist = logger.species_distribution()
    assert isinstance(dist, pd.DataFrame)
    assert list(dist.columns) == ["species", "count"]
    assert len(dist) == 3

    # Zebra has 3, elephant has 2, bird has 1
    assert dist.iloc[0]["species"] == "zebra"
    assert dist.iloc[0]["count"] == 3
    assert dist.iloc[1]["species"] == "elephant"
    assert dist.iloc[1]["count"] == 2
    assert dist.iloc[2]["species"] == "bird"
    assert dist.iloc[2]["count"] == 1


def test_logger_peak_activity_datetime_window() -> None:
    """Verify peak_activity identifies busiest time windows using datetime resampling."""
    logger = DetectionLogger()

    # Minute 0: 5 detections
    for i in range(5):
        logger.log(
            frame_no=i,
            detections=[{"track_id": i, "species": "zebra", "confidence": 0.9}],
            timestamp=f"2026-10-10T12:00:{i:02d}Z",
        )

    # Minute 1: 1 detection
    logger.log(
        frame_no=10,
        detections=[{"track_id": 10, "species": "bird", "confidence": 0.8}],
        timestamp="2026-10-10T12:01:15Z",
    )

    # Minute 2: 12 detections (busiest window)
    for i in range(12):
        logger.log(
            frame_no=20 + i,
            detections=[{"track_id": 100 + i, "species": "elephant", "confidence": 0.9}],
            timestamp=f"2026-10-10T12:02:{i * 2:02d}Z",
        )

    peak = logger.peak_activity(window="1min")
    assert isinstance(peak, pd.DataFrame)
    assert list(peak.columns) == ["window", "count"]
    assert len(peak) == 3

    # Top row must be Minute 2 with count 12
    assert peak.iloc[0]["count"] == 12
    assert "12:02" in str(peak.iloc[0]["window"])

    # Second row must be Minute 0 with count 5
    assert peak.iloc[1]["count"] == 5
    assert "12:00" in str(peak.iloc[1]["window"])


def test_logger_peak_activity_numeric_window() -> None:
    """Verify peak_activity handles numeric frame or second windows."""
    logger = DetectionLogger()
    # 4 detections in frames 0..9, 8 detections in frames 10..19
    for f in range(4):
        logger.log(frame_no=f, detections=[{"track_id": f, "species": "dog", "confidence": 0.9}], timestamp=float(f))
    for f in range(10, 18):
        logger.log(frame_no=f, detections=[{"track_id": f, "species": "cat", "confidence": 0.9}], timestamp=float(f))

    peak = logger.peak_activity(window=10)
    assert len(peak) >= 2
    assert peak.iloc[0]["count"] == 8
    assert peak.iloc[1]["count"] == 4


def test_logger_empty_state_safety() -> None:
    """Verify logger methods execute gracefully without errors when no records exist."""
    logger = DetectionLogger()
    assert logger.dataframe.empty
    assert list(logger.dataframe.columns) == DetectionLogger.COLUMNS

    dist = logger.species_distribution()
    assert dist.empty
    assert list(dist.columns) == ["species", "count"]

    peak = logger.peak_activity(window="1min")
    assert peak.empty
    assert list(peak.columns) == ["window", "count"]
