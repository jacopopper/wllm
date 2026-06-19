from __future__ import annotations

import pytest

from extractors.collectors import CollectorRegistry


def test_collector_cleanup_after_exception() -> None:
    registry = CollectorRegistry()
    with pytest.raises(RuntimeError):
        with registry.scope("req1"):
            registry.record("req1", "value")
            raise RuntimeError("boom")
    assert registry.active_ids() == set()


def test_collector_request_isolation() -> None:
    registry = CollectorRegistry()
    with registry.scope("req1") as req1, registry.scope("req2") as req2:
        assert registry.record("req1", "a")
        assert registry.record("req2", "b")
        assert req1.records == ["a"]
        assert req2.records == ["b"]
    assert not registry.record("req1", "late")
