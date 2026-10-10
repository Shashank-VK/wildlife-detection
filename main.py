"""Command-line interface for the Wildlife Detection and Live Count System.

Processes video files or live webcam streams, displays real-time per-species
counts and tracking events, and exports detection analytics to CSV.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from counter import LiveCounter
from detector import WildlifeDetector, draw_detections
from logger import DetectionLogger


def parse_source(source_str: str) -> int | str:
    """Parse source string into a webcam device index or video file path.

    Args:
        source_str: String input representing a camera index ('0') or file path.

    Returns:
        Integer device index if numeric or 'webcam', otherwise original string path.
    """
    clean_str = source_str.strip()
    if clean_str.lower() in {"webcam", "camera"}:
        return 0
    if clean_str.isdigit():
        return int(clean_str)
    return clean_str


def run_pipeline(
    source: int | str,
    output_path: str | Path | None = "detections.csv",
    detector: WildlifeDetector | None = None,
    counter: LiveCounter | None = None,
    logger: DetectionLogger | None = None,
    show: bool = False,
    max_frames: int | None = None,
    log_interval: int = 1,
) -> tuple[LiveCounter, DetectionLogger]:
    """Execute the wildlife detection, tracking, counting, and logging pipeline.

    Args:
        source: Video file path or integer webcam device index.
        output_path: Optional CSV output path to save detections.
        detector: WildlifeDetector instance (injected or default).
        counter: LiveCounter instance (injected or default).
        logger: DetectionLogger instance (injected or default).
        show: Whether to render an OpenCV display window.
        max_frames: Maximum number of frames to process before stopping.
        log_interval: Interval (in frames) at which to print console statistics.

    Returns:
        Tuple of (counter, logger) instances.
    """
    if detector is None:
        detector = WildlifeDetector()
    if counter is None:
        counter = LiveCounter()
    if logger is None:
        logger = DetectionLogger()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source: {source}", file=sys.stderr)
        return counter, logger

    frame_idx = 0
    print(f"\n🐾 Starting Wildlife Detection Pipeline on source: {source}")
    print("=" * 65)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            now_iso = datetime.now(timezone.utc).isoformat()

            # Run detection and ByteTrack tracking
            detections = detector.detect_and_track(frame, persist=True)

            # Update live counter and record detection logs
            snapshot = counter.update(detections, frame_no=frame_idx, timestamp=now_iso)
            logger.log(frame_no=frame_idx, detections=detections, timestamp=now_iso)

            # Print live counts according to interval
            if frame_idx % log_interval == 0 or snapshot["recent_events"]:
                current_str = ", ".join(
                    f"{sp}: {cnt}" for sp, cnt in snapshot["current_counts"].items()
                ) or "None"
                unique_str = ", ".join(
                    f"{sp}: {cnt}" for sp, cnt in snapshot["cumulative_counts"].items()
                ) or "None"

                print(
                    f"[Frame {frame_idx:05d}] "
                    f"In-Frame: [{current_str}] (Total: {snapshot['total_current']}) | "
                    f"Unique Seen: [{unique_str}] (Total: {snapshot['total_unique']})"
                )

                # Report any track entrance/exit events
                for event in snapshot["recent_events"]:
                    symbol = "🟢 ENTER" if event["event"] == "entered" else "🔴 EXIT "
                    print(
                        f"   └─ {symbol}: #{event['track_id']} "
                        f"{event['species'].capitalize()} (Frame {frame_idx})"
                    )

            # Render display window if requested
            if show:
                annotated = draw_detections(frame, detections)
                cv2.imshow("Wildlife Detection & Tracking", annotated)
                # Break on 'q' or ESC key
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    print("\nUser requested exit via display window.")
                    break

            if max_frames is not None and frame_idx >= max_frames:
                print(f"\nReached maximum frame limit ({max_frames}). Stopping.")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        cap.release()
        if show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    print("=" * 65)
    print(f"✅ Processing complete: {frame_idx} frames processed.")
    final_snap = counter.get_snapshot()
    print(f"Total Unique Animals Seen: {final_snap['total_unique']}")
    for sp, cnt in final_snap["cumulative_counts"].items():
        print(f"  - {sp.capitalize()}: {cnt} unique")

    # Export to CSV if output_path is provided
    if output_path is not None:
        logger.to_csv(output_path)
        print(f"📁 Detections saved to CSV: {output_path} ({len(logger)} rows logged)")

    # Display peak activity summary
    peak_df = logger.peak_activity(window="1min")
    if not peak_df.empty:
        print("\n📊 Peak Activity Windows:")
        print(peak_df.head(5).to_string(index=False))

    return counter, logger


def build_parser() -> argparse.ArgumentParser:
    """Construct argument parser for CLI options."""
    parser = argparse.ArgumentParser(
        description="Wildlife Detection and Live Count System CLI"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to video file (e.g. video.mp4) or webcam identifier ('webcam' or index '0').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detections.csv",
        help="Destination path for output detections CSV file (default: detections.csv).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display OpenCV video window with live annotations.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or weight name (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for animal detections (default: 0.25).",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config name or YAML path (default: bytetrack.yaml).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before stopping.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="Frame interval for printing live counts to console (default: 1).",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    parsed_source = parse_source(args.source)
    detector = WildlifeDetector(
        model_name=args.model,
        conf_threshold=args.conf,
        tracker=args.tracker,
    )
    counter = LiveCounter()
    detection_logger = DetectionLogger()

    run_pipeline(
        source=parsed_source,
        output_path=args.output,
        detector=detector,
        counter=counter,
        logger=detection_logger,
        show=args.show,
        max_frames=args.max_frames,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
