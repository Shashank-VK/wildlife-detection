"""Tests for CLI arguments, source parsing, and Streamlit app utilities."""

from __future__ import annotations

import numpy as np
import pytest

from app import generate_synthetic_demo_frame
from main import build_parser, parse_source


def test_parse_source_variants() -> None:
    """Verify source strings are converted appropriately to int or str."""
    assert parse_source("0") == 0
    assert parse_source("1") == 1
    assert parse_source("webcam") == 0
    assert parse_source("camera") == 0
    assert parse_source("video.mp4") == "video.mp4"
    assert parse_source("/path/to/safari_feed.avi") == "/path/to/safari_feed.avi"


def test_cli_parser_defaults() -> None:
    """Verify CLI parser has standard defaults."""
    parser = build_parser()
    args = parser.parse_args(["--source", "video.mp4"])

    assert args.source == "video.mp4"
    assert args.output == "detections.csv"
    assert args.model == "yolov8n.pt"
    assert args.conf == 0.25
    assert args.tracker == "bytetrack.yaml"
    assert args.show is False
    assert args.max_frames is None
    assert args.log_interval == 1


def test_cli_parser_custom_options() -> None:
    """Verify custom CLI options are parsed properly."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "--source",
            "0",
            "--output",
            "custom.csv",
            "--show",
            "--conf",
            "0.5",
            "--max-frames",
            "100",
            "--log-interval",
            "5",
        ]
    )

    assert args.source == "0"
    assert args.output == "custom.csv"
    assert args.show is True
    assert args.conf == 0.5
    assert args.max_frames == 100
    assert args.log_interval == 5


def test_synthetic_demo_frame_generation() -> None:
    """Verify synthetic demo frame generator creates valid image array."""
    frame_0 = generate_synthetic_demo_frame(0)
    frame_50 = generate_synthetic_demo_frame(50)

    assert isinstance(frame_0, np.ndarray)
    assert frame_0.shape == (480, 640, 3)
    assert frame_0.dtype == np.uint8

    assert isinstance(frame_50, np.ndarray)
    assert frame_50.shape == (480, 640, 3)
