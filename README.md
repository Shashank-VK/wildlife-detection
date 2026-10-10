# Wildlife Detection and Live Count System

An end-to-end Python system for automated animal detection, persistent multi-object tracking (ByteTrack), live per-species census counting, and detection analytics from video files or live camera feeds.

---

## Architecture Diagram

```text
                               +-----------------------------+
                               | Video Source (File / Webcam)|
                               +--------------+--------------+
                                              |
                                              v (Frame Stream)
                               +-----------------------------+
                               |      WildlifeDetector       |
                               |   (Ultralytics YOLOv8n)     |
                               +--------------+--------------+
                                              |
                                              | Raw Detections + Boxes
                                              v
                               +-----------------------------+
                               |     ByteTrack Tracker       |
                               | (Persistent Object Track ID)|
                               +--------------+--------------+
                                              |
                                              | Filtered Animal Detections
                                              | {track_id, species, conf, bbox}
                                              v
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
       +---------------------------+                     +---------------------------+
       |        LiveCounter        |                     |      DetectionLogger      |
       |---------------------------|                     |---------------------------|
       | - In-Frame Active Counts  |                     | - Pandas DataFrame Store  |
       | - Cumulative Unique Seen  |                     | - Species Distribution    |
       | - Enter / Exit Event Log  |                     | - Peak Activity Resampling|
       | - State Snapshot Dict     |                     | - CSV Export Pipeline     |
       +-------------+-------------+                     +-------------+-------------+
                     |                                                 |
                     +------------------------+------------------------+
                                              |
                                              v
                              +-------------------------------+
                              |    User Interfaces / Sinks    |
                              |-------------------------------|
                              | 1. Streamlit Dashboard (GUI)  |
                              | 2. CLI Runner (main.py)       |
                              | 3. CSV Dataset Export         |
                              +-------------------------------+
```

---

## Features

- **YOLOv8 Detection with Lazy Loading**: Uses `yolov8n.pt` downloaded and cached locally on first use, with an injectable model interface for testing and offline development.
- **ByteTrack Multi-Object Tracking**: Maintains persistent numeric track IDs across frames to accurately follow individual animals.
- **Strict Animal Filtering**: Restricts detections to animal classes, filtering out non-animal categories (e.g. humans, vehicles, equipment).
- **Live Census Counter**: Tracks current in-frame counts per species, cumulative unique animals seen (avoiding double-counting re-entering animals), and logs entrance/exit events.
- **Pandas Detection Analytics**: Appends per-frame detection metadata, resamples peak activity periods, calculates species distribution, and exports to CSV.
- **Interactive Streamlit Dashboard**: Provides real-time annotated video playback, live count badges, distribution charts, and dynamic CSV downloads.
- **Deterministic Offline Pytest Suite**: Full test coverage using synthetic numpy frames and injected mock models with zero network downloads required.

---

## Supported Animal Species

The default detector filters detections to the 10 animal categories in the pretrained COCO dataset:

1. `bird`
2. `cat`
3. `dog`
4. `horse`
5. `sheep`
6. `cow`
7. `elephant`
8. `bear`
9. `zebra`
10. `giraffe`

Custom subsets of species can be configured via the `target_classes` parameter or through the Streamlit sidebar.

---

## Installation

### Prerequisites
- Python 3.10+
- `pip`

### Setup

```bash
# Clone or navigate to the repository
cd wildlifedetect

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Command-Line Interface (`main.py`)

Process a video file or live webcam:

```bash
# Process a video file and save detections to CSV
python main.py --source safari_feed.mp4 --output detections.csv

# Process live webcam (index 0) with OpenCV display window
python main.py --source webcam --show

# Process a specific camera index with custom confidence threshold
python main.py --source 0 --conf 0.35 --show

# Limit processing to a fixed number of frames (e.g. for batch jobs)
python main.py --source safari_feed.mp4 --max-frames 300 --output batch_out.csv
```

#### CLI Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--source` | `str` | *Required* | Path to video file or webcam index (`0`, `webcam`). |
| `--output` | `str` | `detections.csv` | Destination path for output CSV. |
| `--show` | `flag` | `False` | Display annotated OpenCV feed window (press 'q' to exit). |
| `--model` | `str` | `yolov8n.pt` | Model weights file or path. |
| `--conf` | `float` | `0.25` | Minimum confidence score threshold. |
| `--tracker` | `str` | `bytetrack.yaml` | Ultralytics tracker configuration. |
| `--max-frames` | `int` | `None` | Optional limit on the number of frames to process. |
| `--log-interval` | `int` | `1` | Interval (in frames) for console print statements. |

---

### 2. Streamlit Dashboard (`app.py`)

Launch the web interface:

```bash
streamlit run app.py
```

Dashboard features:
- **Video Source Selector**: Upload MP4/AVI files, connect to live webcam feeds, or run the built-in **Synthetic Demo Generator** without external files.
- **Live Video Player**: Renders bounding boxes, species labels, confidence scores, and persistent track IDs.
- **Census Panel**: Live metrics for current animals in-frame vs. total cumulative unique animals observed.
- **Species Distribution**: Auto-updating bar chart displaying observed species frequency.
- **Live Event Log**: Real-time log of animal arrival (`ENTERED`) and departure (`LEFT`) events.
- **CSV Export**: One-click download button for recorded detection analytics.

---

## Running Tests

All unit and integration tests use synthetic numpy-generated frames and injectable mock models. No weights are downloaded and no webcam devices are required during testing.

```bash
pytest -v
```

---

## Project Structure

```text
wildlifedetect/
├── detector.py             # WildlifeDetector wrapping YOLOv8 + ByteTrack
├── counter.py              # LiveCounter class for real-time census & events
├── logger.py               # DetectionLogger for Pandas analytics and CSV export
├── app.py                  # Streamlit web dashboard
├── main.py                 # Command-line interface
├── requirements.txt        # Package dependencies
├── pytest.ini              # Pytest configuration
├── .gitignore              # Git ignore rules
├── README.md               # Documentation
└── tests/
    ├── conftest.py         # Mock YOLO models & synthetic frame fixtures
    ├── test_detector.py    # Detector lazy loading, filtering, & drawing tests
    ├── test_counter.py     # LiveCounter enter/exit & ID reuse tests
    ├── test_logger.py      # DetectionLogger DataFrame & peak activity tests
    ├── test_tracker_persistence.py # Multi-frame tracker & pipeline tests
    └── test_cli_and_app.py # CLI options & app utilities tests
```

---

## Scope and Limitations

- **Pretrained COCO Species**: The default model weights (`yolov8n.pt`) are trained on the 80 COCO classes, which includes the 10 animal categories listed above. Specialized wildlife species (e.g. leopards, rhinos, badgers) require fine-tuning or custom YOLO model weights.
- **Tracking Quality**: Multi-object tracking accuracy depends on video framerate, motion blur, occlusion, and lighting conditions. Severe camera movement or long occlusions can result in track ID switches.
- **Hardware Acceleration**: The system defaults to CPU execution and automatically utilizes CUDA if a compatible GPU and PyTorch CUDA build are configured.
