from zephon_modal.registry import Registry


class Store(dict):
    def put(self, key, value, skip_if_exists=False):
        self[key] = value


def test_attempt_records_by_launch_and_run():
    registry = Registry(Store())
    registry.record("r1", "L1", "ta-2", {"task_id": "ta-2", "started": "2026-11-10T01:00:00+00:00"})
    registry.record("r1", "L1", "ta-1", {"task_id": "ta-1", "started": "2026-11-10T00:00:00+00:00"})
    registry.record("r1", "L2", "ta-3", {"task_id": "ta-3", "started": "2026-11-10T02:00:00+00:00"})
    registry.record("r2", "L9", "ta-9", {"task_id": "ta-9", "started": "2026-11-10T00:00:00+00:00"})
    assert [r["task_id"] for r in registry.launch_records("r1", "L1")] == ["ta-1", "ta-2"]
    assert [(r["launch_id"], r["task_id"]) for r in registry.run_records("r1")] == [
        ("L1", "ta-1"),
        ("L1", "ta-2"),
        ("L2", "ta-3"),
    ]
    # A heartbeat replaces the container's record rather than adding one.
    registry.record("r1", "L1", "ta-1", {"task_id": "ta-1", "started": "2026-11-10T00:00:00+00:00", "heartbeat": "h"})
    assert len(registry.launch_records("r1", "L1")) == 2
    # Attempt records aren't runs.
    assert registry.run_ids() == []


def test_requests_are_taken_once():
    registry = Registry(Store())
    registry.request("stop", "r1")
    assert registry.take("stop", "r1") is True
    assert registry.take("stop", "r1") is None
