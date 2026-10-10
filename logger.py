"""Wildlife detection logging and analytics module.

Records detection events per frame, maintains an underlying Pandas DataFrame,
and provides analytics for species distributions, peak activity windows, and CSV export.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


class DetectionLogger:
    """Logs per-frame wildlife detections to a Pandas DataFrame and computes analytics."""

    COLUMNS = ["timestamp", "frame_no", "track_id", "species", "confidence", "bbox"]

    def __init__(self, initial_df: pd.DataFrame | None = None) -> None:
        """Initialize the DetectionLogger.

        Args:
            initial_df: Optional initial DataFrame to populate the logger.
        """
        self._records: list[dict[str, Any]] = []
        if initial_df is not None and not initial_df.empty:
            self._records = initial_df.to_dict(orient="records")

    def log(
        self,
        frame_no: int,
        detections: Sequence[dict[str, Any]],
        timestamp: Any = None,
    ) -> None:
        """Append detections for a single frame to the logger.

        Args:
            frame_no: Video frame index.
            detections: List of detection dicts (track_id, species, confidence, bbox).
            timestamp: Frame timestamp (ISO string, datetime, or float). Defaults to current UTC time.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat()
        elif isinstance(timestamp, datetime):
            timestamp = timestamp.isoformat()

        for det in detections:
            track_id = det.get("track_id")
            if track_id is not None:
                try:
                    track_id = int(track_id)
                except (ValueError, TypeError):
                    pass

            row = {
                "timestamp": timestamp,
                "frame_no": int(frame_no),
                "track_id": track_id,
                "species": str(det.get("species", "")).lower().strip(),
                "confidence": round(float(det.get("confidence", 0.0)), 4),
                "bbox": det.get("bbox"),
            }
            self._records.append(row)

    def log_frame(
        self,
        frame_no: int,
        detections: Sequence[dict[str, Any]],
        timestamp: Any = None,
    ) -> None:
        """Alias for log()."""
        self.log(frame_no=frame_no, detections=detections, timestamp=timestamp)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return the logged detections as a Pandas DataFrame."""
        if not self._records:
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.DataFrame(self._records)
        # Ensure all standard columns are present
        for col in self.COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df[self.COLUMNS]

    def to_csv(self, path: str | Path) -> None:
        """Export logged detections to a CSV file.

        Args:
            path: Destination file path for CSV export.
        """
        dest_path = Path(path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataframe.to_csv(dest_path, index=False)

    @classmethod
    def from_csv(cls, path: str | Path) -> DetectionLogger:
        """Load a DetectionLogger from an existing CSV file.

        Args:
            path: Source CSV file path.
        """
        df = pd.read_csv(path)
        return cls(initial_df=df)

    def species_distribution(self) -> pd.DataFrame:
        """Compute the frequency distribution of detections per species.

        Returns:
            DataFrame with columns ['species', 'count'], sorted descending by count.
        """
        df = self.dataframe
        if df.empty or "species" not in df.columns:
            return pd.DataFrame(columns=["species", "count"])

        dist = df["species"].value_counts().reset_index()
        dist.columns = ["species", "count"]
        return dist

    def peak_activity(self, window: str | int | float = "1min") -> pd.DataFrame:
        """Compute the busiest activity periods based on detection counts.

        Supports both datetime-based time windows (e.g. '1min', '10s', '1h')
        and numeric frame or second windows (e.g. 10, 50).

        Args:
            window: Aggregation window ('1min', '10s', or integer/float).

        Returns:
            DataFrame with columns ['window', 'count'], sorted descending by count.
        """
        df = self.dataframe.copy()
        if df.empty:
            return pd.DataFrame(columns=["window", "count"])

        # Attempt datetime-based resampling if window is a string
        if isinstance(window, str):
            try:
                # Convert timestamps to datetime
                df["dt"] = pd.to_datetime(df["timestamp"])
                resampled = df.set_index("dt").resample(window).size().reset_index()
                resampled.columns = ["window", "count"]
                # Keep active windows and sort by busiest
                active = resampled[resampled["count"] > 0].sort_values(
                    by="count", ascending=False
                ).reset_index(drop=True)
                # Format window as ISO string representation for clean display
                active["window"] = active["window"].astype(str)
                return active
            except Exception:
                pass  # Fall back to frame or numeric binning

        # Numeric or frame-based binning
        num_window = float(window) if not isinstance(window, str) else 10.0
        if num_window <= 0:
            num_window = 1.0

        # Try numeric timestamp first, otherwise use frame_no
        numeric_series = None
        if "timestamp" in df.columns:
            try:
                numeric_series = pd.to_numeric(df["timestamp"])
            except (ValueError, TypeError):
                numeric_series = None

        if numeric_series is None or numeric_series.isna().any():
            numeric_series = pd.to_numeric(df["frame_no"], errors="coerce").fillna(0)

        df["bin"] = (numeric_series // num_window) * num_window
        grouped = df.groupby("bin", as_index=False).size()
        grouped.columns = ["window", "count"]
        sorted_grouped = grouped.sort_values(by="count", ascending=False).reset_index(drop=True)
        return sorted_grouped

    def clear(self) -> None:
        """Clear all logged detections."""
        self._records.clear()

    def __len__(self) -> int:
        """Return total number of detection records logged."""
        return len(self._records)
