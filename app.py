"""Streamlit dashboard for Wildlife Detection and Live Count System.

Provides live video display with bounding boxes and track IDs,
live per-species metrics, distribution charts, and CSV export.
"""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from counter import LiveCounter
from detector import DEFAULT_ANIMAL_CLASSES, WildlifeDetector, draw_detections
from logger import DetectionLogger


@st.cache_resource
def get_detector(
    model_name: str,
    conf_threshold: float,
    tracker: str,
    target_classes: tuple[str, ...],
) -> WildlifeDetector:
    """Cache and return WildlifeDetector instance."""
    return WildlifeDetector(
        model_name=model_name,
        conf_threshold=conf_threshold,
        tracker=tracker,
        target_classes=set(target_classes),
    )


def generate_synthetic_demo_frame(frame_no: int) -> np.ndarray:
    """Generate a clean synthetic video frame for demo/preview without camera."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (30, 45, 30)  # dark forest-green background

    # Ground line
    cv2.line(img, (0, 360), (640, 360), (50, 80, 50), 2)

    # Moving synthetic shapes simulating animals
    t = frame_no * 0.05
    x1 = int(120 + 80 * np.sin(t))
    y1 = int(220 + 20 * np.cos(t))
    # Animal 1 (e.g. zebra proxy)
    cv2.rectangle(img, (x1, y1), (x1 + 90, y1 + 70), (180, 180, 180), -1)
    cv2.putText(
        img,
        "DEMO ZEBRA",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Animal 2 (e.g. elephant proxy)
    if frame_no % 120 < 90:  # Enters and exits
        x2 = int(350 + (frame_no % 120) * 2)
        y2 = 240
        cv2.rectangle(img, (x2, y2), (x2 + 130, y2 + 90), (120, 110, 100), -1)
        cv2.putText(
            img,
            "DEMO ELEPHANT",
            (x2, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        img,
        f"Synthetic Stream - Frame {frame_no}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (100, 200, 100),
        2,
    )
    return img


def main() -> None:
    st.set_page_config(
        page_title="Wildlife Detection & Live Count",
        page_icon="🐾",
        layout="wide",
    )

    st.title("🐾 Wildlife Detection & Live Count Dashboard")
    st.caption(
        "Real-time wildlife detection, ByteTrack multi-object tracking, and automated census system."
    )

    # Initialize session state for persistent counting and logging
    if "counter" not in st.session_state:
        st.session_state["counter"] = LiveCounter()
    if "logger" not in st.session_state:
        st.session_state["logger"] = DetectionLogger()
    if "is_running" not in st.session_state:
        st.session_state["is_running"] = False

    counter: LiveCounter = st.session_state["counter"]
    det_logger: DetectionLogger = st.session_state["logger"]

    # --- SIDEBAR CONFIGURATION ---
    st.sidebar.header("⚙️ Configuration")

    source_type = st.sidebar.selectbox(
        "Video Source",
        ["Video File Upload", "Webcam Device", "Synthetic Demo Generator"],
        index=0,
    )

    temp_video_path: str | None = None
    webcam_index = 0

    if source_type == "Video File Upload":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video File",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            temp_video_path = tfile.name
    elif source_type == "Webcam Device":
        webcam_index = int(
            st.sidebar.number_input("Webcam Device Index", min_value=0, max_value=10, value=0)
        )

    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    target_species = st.sidebar.multiselect(
        "Target Animal Species",
        options=sorted(list(DEFAULT_ANIMAL_CLASSES)),
        default=sorted(list(DEFAULT_ANIMAL_CLASSES)),
    )

    # --- CONTROLS ---
    st.sidebar.subheader("🕹️ Controls")
    ctrl_col1, ctrl_col2 = st.sidebar.columns(2)
    start_clicked = ctrl_col1.button("▶ Start", use_container_width=True)
    stop_clicked = ctrl_col2.button("⏹ Stop", use_container_width=True)

    if st.sidebar.button("🔄 Reset Counts & Logs", use_container_width=True):
        counter.reset()
        det_logger.clear()
        st.session_state["is_running"] = False
        st.sidebar.success("Counters and detection logs reset.")

    if start_clicked:
        st.session_state["is_running"] = True
    if stop_clicked:
        st.session_state["is_running"] = False

    # --- SIDEBAR METRICS ---
    st.sidebar.subheader("📊 Live Census Snapshot")
    snap = counter.get_snapshot()

    m_col1, m_col2 = st.sidebar.columns(2)
    m_col1.metric("In-Frame Now", snap["total_current"])
    m_col2.metric("Total Unique", snap["total_unique"])

    # Species breakdown table in sidebar
    if snap["cumulative_counts"] or snap["current_counts"]:
        all_sp = sorted(set(snap["cumulative_counts"].keys()) | set(snap["current_counts"].keys()))
        breakdown_data = [
            {
                "Species": sp.capitalize(),
                "In-Frame": snap["current_counts"].get(sp, 0),
                "Unique Seen": snap["cumulative_counts"].get(sp, 0),
            }
            for sp in all_sp
        ]
        st.sidebar.dataframe(
            pd.DataFrame(breakdown_data),
            use_container_width=True,
            hide_index=True,
        )

    # --- MAIN CONTENT LAYOUT ---
    feed_col, stats_col = st.columns([3, 2])

    with feed_col:
        st.subheader("Live Video Stream")
        feed_placeholder = st.empty()
        status_placeholder = st.empty()

    with stats_col:
        st.subheader("Species Distribution")
        chart_placeholder = st.empty()

        st.subheader("Recent Tracking Events")
        events_placeholder = st.empty()

        st.subheader("Export Data")
        export_placeholder = st.empty()

    def update_analytics_view() -> None:
        """Render charts, events list, and CSV download button."""
        # Species distribution chart
        dist_df = det_logger.species_distribution()
        if not dist_df.empty:
            chart_placeholder.bar_chart(
                data=dist_df.set_index("species")["count"],
                color="#2ecc71",
            )
        else:
            chart_placeholder.info("No detections recorded yet.")

        # Recent events list
        current_snap = counter.get_snapshot()
        all_events = current_snap.get("events", [])
        if all_events:
            recent_display = []
            for ev in all_events[-8:][::-1]:
                action = "🟢 ENTERED" if ev["event"] == "entered" else "🔴 LEFT"
                recent_display.append(
                    {
                        "Event": action,
                        "Track ID": f"#{ev['track_id']}",
                        "Species": ev["species"].capitalize(),
                        "Frame": ev["frame_no"],
                    }
                )
            events_placeholder.dataframe(
                pd.DataFrame(recent_display),
                use_container_width=True,
                hide_index=True,
            )
        else:
            events_placeholder.info("No entrance/exit events recorded yet.")

        # CSV Download button
        csv_data = det_logger.dataframe.to_csv(index=False)
        export_placeholder.download_button(
            label="📥 Download Detections CSV",
            data=csv_data,
            file_name=f"wildlife_detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Initial analytics display
    update_analytics_view()

    # --- VIDEO PROCESSING LOOP ---
    if st.session_state["is_running"]:
        if source_type == "Video File Upload" and temp_video_path is None:
            status_placeholder.warning("Please upload a video file to start.")
            st.session_state["is_running"] = False
            return

        # Instantiate or retrieve cached detector
        detector = get_detector(
            model_name="yolov8n.pt",
            conf_threshold=conf_threshold,
            tracker="bytetrack.yaml",
            target_classes=tuple(target_species),
        )

        cap: cv2.VideoCapture | None = None
        if source_type == "Video File Upload" and temp_video_path is not None:
            cap = cv2.VideoCapture(temp_video_path)
        elif source_type == "Webcam Device":
            cap = cv2.VideoCapture(webcam_index)

        frame_count = 0
        status_placeholder.info("Detection loop running...")

        try:
            while st.session_state["is_running"]:
                frame = None
                if source_type == "Synthetic Demo Generator":
                    frame = generate_synthetic_demo_frame(frame_count)
                    time.sleep(0.04)  # simulate ~25 FPS
                elif cap is not None:
                    ret, raw_frame = cap.read()
                    if not ret or raw_frame is None:
                        status_placeholder.success("Video playback finished.")
                        st.session_state["is_running"] = False
                        break
                    frame = raw_frame

                if frame is None:
                    break

                frame_count += 1
                now_str = datetime.now(timezone.utc).isoformat()

                # In synthetic demo mode, inject synthetic detections for visualization if model isn't active
                if source_type == "Synthetic Demo Generator":
                    t = frame_count * 0.05
                    x1 = 120 + 80 * np.sin(t)
                    y1 = 220 + 20 * np.cos(t)
                    synth_detections = [
                        {
                            "track_id": 1,
                            "species": "zebra",
                            "confidence": 0.92,
                            "bbox": [x1, y1, x1 + 90, y1 + 70],
                        }
                    ]
                    if frame_count % 120 < 90:
                        x2 = 350 + (frame_count % 120) * 2
                        synth_detections.append(
                            {
                                "track_id": 2,
                                "species": "elephant",
                                "confidence": 0.88,
                                "bbox": [x2, 240, x2 + 130, 330],
                            }
                        )
                    detections = synth_detections
                else:
                    detections = detector.detect_and_track(frame, persist=True)

                # Update state
                counter.update(detections, frame_no=frame_count, timestamp=now_str)
                det_logger.log(frame_no=frame_count, detections=detections, timestamp=now_str)

                # Annotate and show image
                annotated = draw_detections(frame, detections)
                rgb_img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(rgb_img, channels="RGB", use_container_width=True)

                # Refresh metrics and distribution periodically
                if frame_count % 5 == 0:
                    update_analytics_view()

        finally:
            if cap is not None:
                cap.release()

        update_analytics_view()


if __name__ == "__main__":
    main()
