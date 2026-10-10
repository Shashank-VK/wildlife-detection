"""Wildlife detector module using Ultralytics YOLOv8 with ByteTrack tracking.

This module provides the WildlifeDetector class which wraps YOLOv8 detection and
multi-object tracking (ByteTrack), filtering outputs strictly to animal classes.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Standard 10 animal classes present in the pretrained COCO dataset for YOLOv8
DEFAULT_ANIMAL_CLASSES: set[str] = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


class WildlifeDetector:
    """YOLOv8 wrapper for animal detection and persistent multi-object tracking.

    Supports lazy loading of the underlying YOLO model and dependency injection
    for testing with mock/stub models.
    """

    def __init__(
        self,
        model: Any = None,
        model_name: str = "yolov8n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        tracker: str = "bytetrack.yaml",
        target_classes: Iterable[str] | None = None,
        device: str | None = None,
    ) -> None:
        """Initialize the WildlifeDetector.

        Args:
            model: Pre-instantiated or mock model. If None, lazy-loads on first use.
            model_name: Name or path of the YOLO model weights (default 'yolov8n.pt').
            conf_threshold: Minimum detection confidence threshold (0.0 to 1.0).
            iou_threshold: IoU threshold for NMS.
            tracker: Tracker configuration filename or path (default 'bytetrack.yaml').
            target_classes: Set of allowed species names. Defaults to the 10 COCO animal classes.
            device: Computing device ('cpu', 'cuda', etc.). If None, automatically selected.
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker = tracker
        self.device = device

        if target_classes is None:
            self.target_classes = set(DEFAULT_ANIMAL_CLASSES)
        else:
            self.target_classes = {cls.lower().strip() for cls in target_classes}

        self._model = model

    @property
    def model(self) -> Any:
        """Retrieve the underlying model, lazy-loading it if necessary."""
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self) -> None:
        """Explicitly load the YOLO model weights."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        logger.info("Loading YOLO model: %s", self.model_name)
        self._model = YOLO(self.model_name)

    def detect_and_track(
        self,
        frame: np.ndarray,
        persist: bool = True,
    ) -> list[dict[str, Any]]:
        """Run detection and tracking on a single image frame.

        Args:
            frame: Input image as a NumPy array (BGR or RGB).
            persist: Whether to persist track IDs across consecutive frames.

        Returns:
            List of detection dicts, each containing:
                - 'track_id': int | None (persistent object track identifier)
                - 'species': str (detected animal species name)
                - 'confidence': float (detection confidence between 0.0 and 1.0)
                - 'bbox': list[float] ([x1, y1, x2, y2] bounding box coordinates)
        """
        model = self.model

        # Execute tracking using ultralytics built-in tracker
        track_kwargs: dict[str, Any] = {
            "source": frame,
            "persist": persist,
            "tracker": self.tracker,
            "conf": self.conf_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
        }
        if self.device is not None:
            track_kwargs["device"] = self.device

        results = model.track(**track_kwargs)
        if not results:
            return []

        result = results[0]
        return self._parse_result(result)

    def _parse_result(self, result: Any) -> list[dict[str, Any]]:
        """Extract and filter animal detections from a model result object.

        Args:
            result: Ultralytics Results object or duck-typed test stub.

        Returns:
            Filtered list of detection dictionaries for animal classes.
        """
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        names: dict[int, str] = getattr(result, "names", {})

        # Extract tensors / arrays from boxes
        xyxy_data = getattr(boxes, "xyxy", None)
        conf_data = getattr(boxes, "conf", None)
        cls_data = getattr(boxes, "cls", None)
        id_data = getattr(boxes, "id", None)

        detections: list[dict[str, Any]] = []
        num_boxes = len(boxes)

        for i in range(num_boxes):
            # Resolve class ID and species name
            if cls_data is not None:
                cls_val = cls_data[i]
                cls_idx = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
            else:
                cls_idx = 0

            species = names.get(cls_idx, str(cls_idx)).lower().strip()

            # Filter to animal classes only
            if species not in self.target_classes:
                continue

            # Resolve confidence
            if conf_data is not None:
                conf_val = conf_data[i]
                conf = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)
            else:
                conf = 1.0

            if conf < self.conf_threshold:
                continue

            # Resolve track ID
            track_id: int | None = None
            if id_data is not None:
                tid_val = id_data[i]
                if tid_val is not None:
                    try:
                        raw_tid = tid_val.item() if hasattr(tid_val, "item") else tid_val
                        track_id = int(raw_tid)
                    except (ValueError, TypeError):
                        track_id = None

            # Resolve bounding box coordinates [x1, y1, x2, y2]
            if xyxy_data is not None:
                box_vals = xyxy_data[i]
                if hasattr(box_vals, "tolist"):
                    bbox = [round(float(coord), 2) for coord in box_vals.tolist()]
                else:
                    bbox = [round(float(coord), 2) for coord in box_vals]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]

            detections.append(
                {
                    "track_id": track_id,
                    "species": species,
                    "confidence": round(conf, 4),
                    "bbox": bbox,
                }
            )

        return detections


def draw_detections(
    frame: np.ndarray,
    detections: Sequence[dict[str, Any]],
    box_color: tuple[int, int, int] = (0, 255, 0),
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_thickness: int = 2,
) -> np.ndarray:
    """Draw bounding boxes and metadata labels onto a video frame.

    Args:
        frame: Original frame as NumPy array (BGR or RGB).
        detections: List of detection dicts with 'bbox', 'species', 'confidence', 'track_id'.
        box_color: Bounding box BGR/RGB tuple.
        text_color: Label text color.
        line_thickness: Width of the bounding box borders.

    Returns:
        Annotated frame as a copy of the input array.
    """
    annotated = frame.copy()
    height, width = annotated.shape[:2]

    # Pre-generate distinct colors for different track IDs
    color_palette = [
        (46, 204, 113),  # Emerald Green
        (52, 152, 219),  # Peter River Blue
        (155, 89, 182),  # Amethyst Purple
        (241, 196, 15),  # Sunflower Yellow
        (230, 126, 34),  # Carrot Orange
        (231, 76, 60),   # Alizarin Red
        (26, 188, 156),  # Turquoise
        (243, 156, 18),  # Orange
    ]

    for det in detections:
        bbox = det.get("bbox", [0, 0, 0, 0])
        x1 = max(0, min(int(bbox[0]), width - 1))
        y1 = max(0, min(int(bbox[1]), height - 1))
        x2 = max(0, min(int(bbox[2]), width - 1))
        y2 = max(0, min(int(bbox[3]), height - 1))

        track_id = det.get("track_id")
        species = det.get("species", "animal")
        conf = det.get("confidence", 0.0)

        # Pick color based on track_id if available
        if track_id is not None:
            color = color_palette[track_id % len(color_palette)]
            label = f"#{track_id} {species.capitalize()} ({conf:.2f})"
        else:
            color = box_color
            label = f"{species.capitalize()} ({conf:.2f})"

        # Draw bounding box rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Draw filled background for label text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        bg_y1 = max(0, y1 - label_h - baseline - 4)
        bg_y2 = y1
        bg_x2 = min(width, x1 + label_w + 6)

        cv2.rectangle(annotated, (x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 3, bg_y2 - baseline),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return annotated
