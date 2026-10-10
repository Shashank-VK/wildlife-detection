"""Live wildlife counting module for tracking animals across video frames.

Maintains real-time per-species counts, cumulative unique counts based on persistent
track IDs, and records entrance/exit events as animals appear or disappear.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence


class LiveCounter:
    """Maintains real-time and cumulative animal counts and tracks entrance/exit events."""

    def __init__(self) -> None:
        """Initialize empty state for the LiveCounter."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracking and counting state to initial values."""
        self.current_frame: int = 0
        self.current_timestamp: Any = None

        # Map of currently visible track_id -> species
        self._active_tracks: dict[int, str] = {}

        # Set of unique track IDs seen per species
        self._seen_tracks_by_species: defaultdict[str, set[int]] = defaultdict(set)

        # Set of all unique track IDs seen across all species
        self._all_seen_tracks: set[int] = set()

        # Detections without track ID in the current frame
        self._untracked_current: defaultdict[str, int] = defaultdict(int)

        # Cumulative count of untracked animals (fallback when track_id is None)
        self._untracked_cumulative: defaultdict[str, int] = defaultdict(int)

        # Historical event log: list of dicts {event, track_id, species, frame_no, timestamp}
        self._events: list[dict[str, Any]] = []

        # Events generated during the most recent update()
        self._recent_events: list[dict[str, Any]] = []

        # Current frame counts per species
        self._current_counts: dict[str, int] = {}

        # Cumulative unique counts per species
        self._cumulative_counts: dict[str, int] = {}

    def update(
        self,
        detections: Sequence[dict[str, Any]],
        frame_no: int = 0,
        timestamp: Any = None,
    ) -> dict[str, Any]:
        """Update counts and detect entrance/exit events for a new frame.

        Args:
            detections: List of detection dicts with 'track_id', 'species', etc.
            frame_no: Current frame sequence index.
            timestamp: Optional timestamp for the frame.

        Returns:
            Snapshot dictionary containing counts, active tracks, and recent events.
        """
        self.current_frame = frame_no
        self.current_timestamp = timestamp
        self._recent_events = []

        current_active_tracks: dict[int, str] = {}
        frame_species_counts: defaultdict[str, int] = defaultdict(int)
        untracked_in_frame: defaultdict[str, int] = defaultdict(int)

        for det in detections:
            species = str(det.get("species", "unknown")).lower().strip()
            track_id = det.get("track_id")

            frame_species_counts[species] += 1

            if track_id is not None:
                tid = int(track_id)
                current_active_tracks[tid] = species
                self._seen_tracks_by_species[species].add(tid)
                self._all_seen_tracks.add(tid)
            else:
                untracked_in_frame[species] += 1
                self._untracked_cumulative[species] += 1

        self._untracked_current = untracked_in_frame

        # Detect entering tracks: newly appeared in this frame
        prev_track_ids = set(self._active_tracks.keys())
        curr_track_ids = set(current_active_tracks.keys())

        entered_ids = curr_track_ids - prev_track_ids
        for tid in sorted(entered_ids):
            species = current_active_tracks[tid]
            event = {
                "event": "entered",
                "track_id": tid,
                "species": species,
                "frame_no": frame_no,
                "timestamp": timestamp,
            }
            self._recent_events.append(event)
            self._events.append(event)

        # Detect leaving tracks: disappeared from this frame
        left_ids = prev_track_ids - curr_track_ids
        for tid in sorted(left_ids):
            species = self._active_tracks[tid]
            event = {
                "event": "left",
                "track_id": tid,
                "species": species,
                "frame_no": frame_no,
                "timestamp": timestamp,
            }
            self._recent_events.append(event)
            self._events.append(event)

        # Update active tracks
        self._active_tracks = current_active_tracks
        self._current_counts = dict(frame_species_counts)

        # Calculate cumulative unique counts per species
        # If tracks are used, count unique track IDs; if no track IDs were ever used, use untracked counts
        all_species = set(self._seen_tracks_by_species.keys()) | set(self._untracked_cumulative.keys())
        cum_counts: dict[str, int] = {}
        for sp in all_species:
            unique_tracked = len(self._seen_tracks_by_species[sp])
            untracked = self._untracked_cumulative[sp]
            cum_counts[sp] = unique_tracked if unique_tracked > 0 else untracked
        self._cumulative_counts = cum_counts

        return self.get_snapshot()

    def get_snapshot(self) -> dict[str, Any]:
        """Expose current state snapshot dictionary."""
        total_unique = (
            len(self._all_seen_tracks)
            if self._all_seen_tracks
            else sum(self._untracked_cumulative.values())
        )

        return {
            "frame_no": self.current_frame,
            "timestamp": self.current_timestamp,
            "current_counts": dict(self._current_counts),
            "cumulative_counts": dict(self._cumulative_counts),
            "total_current": sum(self._current_counts.values()),
            "total_unique": total_unique,
            "active_tracks": dict(self._active_tracks),
            "recent_events": list(self._recent_events),
            "events": list(self._events),
        }

    @property
    def snapshot(self) -> dict[str, Any]:
        """Property shortcut for get_snapshot()."""
        return self.get_snapshot()

    def current_count(self, species: str | None = None) -> int:
        """Return the current in-frame count for a specific species or all animals.

        Args:
            species: Species name (e.g. 'zebra'). If None, returns total across all species.
        """
        if species is None:
            return sum(self._current_counts.values())
        return self._current_counts.get(species.lower().strip(), 0)

    def cumulative_count(self, species: str | None = None) -> int:
        """Return cumulative unique count seen for a specific species or all animals.

        Args:
            species: Species name (e.g. 'zebra'). If None, returns total unique seen.
        """
        if species is None:
            return (
                len(self._all_seen_tracks)
                if self._all_seen_tracks
                else sum(self._untracked_cumulative.values())
            )
        return self._cumulative_counts.get(species.lower().strip(), 0)

    def get_events(
        self,
        event_type: str | None = None,
        species: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query recorded events by type ('entered' / 'left') and/or species.

        Args:
            event_type: Optional filter ('entered' or 'left').
            species: Optional filter by species name.
        """
        results = self._events
        if event_type is not None:
            norm_event = event_type.lower().strip()
            results = [e for e in results if e["event"] == norm_event]
        if species is not None:
            norm_species = species.lower().strip()
            results = [e for e in results if e["species"] == norm_species]
        return results
