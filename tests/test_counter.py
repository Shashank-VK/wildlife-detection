"""Tests for the LiveCounter module covering counts, events, and track ID reuse."""

from __future__ import annotations

import pytest

from counter import LiveCounter


def test_counter_initial_state() -> None:
    """Verify counter initializes with zero counts and empty lists."""
    counter = LiveCounter()
    snap = counter.get_snapshot()

    assert snap["frame_no"] == 0
    assert snap["total_current"] == 0
    assert snap["total_unique"] == 0
    assert snap["current_counts"] == {}
    assert snap["cumulative_counts"] == {}
    assert snap["events"] == []
    assert snap["recent_events"] == []
    assert snap["active_tracks"] == {}


def test_counter_entering_events() -> None:
    """Verify newly detected track IDs generate 'entered' events."""
    counter = LiveCounter()
    detections_frame_1 = [
        {"track_id": 1, "species": "zebra", "confidence": 0.9, "bbox": [10, 10, 50, 50]},
        {"track_id": 2, "species": "elephant", "confidence": 0.85, "bbox": [100, 100, 150, 150]},
    ]
    snap = counter.update(detections_frame_1, frame_no=1)

    assert snap["total_current"] == 2
    assert snap["total_unique"] == 2
    assert snap["current_counts"] == {"zebra": 1, "elephant": 1}
    assert snap["cumulative_counts"] == {"zebra": 1, "elephant": 1}

    # Should have two entered events
    assert len(snap["recent_events"]) == 2
    events = snap["recent_events"]
    assert all(e["event"] == "entered" for e in events)
    assert {e["track_id"] for e in events} == {1, 2}


def test_counter_leaving_events() -> None:
    """Verify disappearing track IDs generate 'left' events."""
    counter = LiveCounter()
    # Frame 1: track 1 and track 2 present
    counter.update(
        [
            {"track_id": 1, "species": "zebra", "confidence": 0.9},
            {"track_id": 2, "species": "elephant", "confidence": 0.9},
        ],
        frame_no=1,
    )

    # Frame 2: only track 1 remains (track 2 left)
    snap = counter.update(
        [{"track_id": 1, "species": "zebra", "confidence": 0.9}],
        frame_no=2,
    )

    assert snap["total_current"] == 1
    assert snap["total_unique"] == 2  # Total unique seen remains 2
    assert snap["current_counts"] == {"zebra": 1}
    assert snap["cumulative_counts"] == {"zebra": 1, "elephant": 1}

    # Recent events should capture track 2 leaving
    assert len(snap["recent_events"]) == 1
    leave_event = snap["recent_events"][0]
    assert leave_event["event"] == "left"
    assert leave_event["track_id"] == 2
    assert leave_event["species"] == "elephant"
    assert leave_event["frame_no"] == 2


def test_counter_id_reuse_across_frames() -> None:
    """Verify persistent ID across frames is not double-counted even after leaving and returning."""
    counter = LiveCounter()

    # Frame 1: track 5 (zebra) enters
    snap1 = counter.update([{"track_id": 5, "species": "zebra"}], frame_no=1)
    assert snap1["total_current"] == 1
    assert snap1["total_unique"] == 1
    assert len(snap1["recent_events"]) == 1
    assert snap1["recent_events"][0]["event"] == "entered"

    # Frame 2: track 5 persists (no enter/leave event)
    snap2 = counter.update([{"track_id": 5, "species": "zebra"}], frame_no=2)
    assert snap2["total_current"] == 1
    assert snap2["total_unique"] == 1
    assert len(snap2["recent_events"]) == 0

    # Frame 3: track 5 disappears (left)
    snap3 = counter.update([], frame_no=3)
    assert snap3["total_current"] == 0
    assert snap3["total_unique"] == 1
    assert len(snap3["recent_events"]) == 1
    assert snap3["recent_events"][0]["event"] == "left"
    assert snap3["recent_events"][0]["track_id"] == 5

    # Frame 4: track 5 reappears (re-enters frame)
    snap4 = counter.update([{"track_id": 5, "species": "zebra"}], frame_no=4)
    assert snap4["total_current"] == 1
    # CRITICAL: cumulative unique animals must still be 1, NOT 2
    assert snap4["total_unique"] == 1
    assert counter.cumulative_count("zebra") == 1
    assert len(snap4["recent_events"]) == 1
    assert snap4["recent_events"][0]["event"] == "entered"


def test_counter_multiple_species_distribution() -> None:
    """Verify counting across multiple different species simultaneously."""
    counter = LiveCounter()

    # Frame 1: 2 zebras, 1 giraffe, 3 birds
    frame_dets = [
        {"track_id": 1, "species": "zebra"},
        {"track_id": 2, "species": "zebra"},
        {"track_id": 3, "species": "giraffe"},
        {"track_id": 4, "species": "bird"},
        {"track_id": 5, "species": "bird"},
        {"track_id": 6, "species": "bird"},
    ]
    snap = counter.update(frame_dets, frame_no=1)

    assert snap["total_current"] == 6
    assert snap["total_unique"] == 6
    assert counter.current_count("zebra") == 2
    assert counter.current_count("giraffe") == 1
    assert counter.current_count("bird") == 3
    assert counter.current_count("bear") == 0

    # Cumulative queries
    assert counter.cumulative_count("zebra") == 2
    assert counter.cumulative_count("giraffe") == 1
    assert counter.cumulative_count("bird") == 3
    assert counter.cumulative_count() == 6


def test_counter_untracked_detections_fallback() -> None:
    """Verify detections without track_id (None) are counted in in-frame totals."""
    counter = LiveCounter()

    # Frame with untracked animals
    untracked_dets = [
        {"track_id": None, "species": "bear"},
        {"track_id": None, "species": "bear"},
    ]
    snap = counter.update(untracked_dets, frame_no=1)

    assert snap["total_current"] == 2
    assert snap["current_counts"]["bear"] == 2
    assert counter.current_count("bear") == 2


def test_counter_get_events_query() -> None:
    """Verify querying historical events by type and species."""
    counter = LiveCounter()
    counter.update([{"track_id": 1, "species": "cat"}, {"track_id": 2, "species": "dog"}], frame_no=1)
    counter.update([{"track_id": 1, "species": "cat"}], frame_no=2)  # dog left

    entered_events = counter.get_events(event_type="entered")
    assert len(entered_events) == 2

    left_events = counter.get_events(event_type="left")
    assert len(left_events) == 1
    assert left_events[0]["species"] == "dog"

    dog_events = counter.get_events(species="dog")
    assert len(dog_events) == 2  # entered and left


def test_counter_reset() -> None:
    """Verify reset wipes all counters, tracking state, and events."""
    counter = LiveCounter()
    counter.update([{"track_id": 1, "species": "horse"}], frame_no=1)
    assert counter.cumulative_count() == 1

    counter.reset()
    assert counter.cumulative_count() == 0
    assert counter.current_count() == 0
    assert len(counter.get_events()) == 0
    assert counter.current_frame == 0
