from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator


@dataclass
class RequestCollector:
    request_id: str
    records: list[Any] = field(default_factory=list)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def record(self, value: Any) -> None:
        with self._lock:
            self.records.append(value)

    def snapshot(self) -> list[Any]:
        with self._lock:
            return list(self.records)


class CollectorRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._collectors: dict[str, RequestCollector] = {}

    @contextmanager
    def scope(self, request_id: str) -> Iterator[RequestCollector]:
        collector = RequestCollector(request_id=request_id)
        with self._lock:
            if request_id in self._collectors:
                raise ValueError(f"collector already exists for request {request_id}")
            self._collectors[request_id] = collector
        try:
            yield collector
        finally:
            with self._lock:
                self._collectors.pop(request_id, None)

    def record(self, request_id: str, value: Any) -> bool:
        with self._lock:
            collector = self._collectors.get(request_id)
            if collector is None:
                return False
            collector.record(value)
            return True

    def active_ids(self) -> set[str]:
        with self._lock:
            return set(self._collectors)

    def get(self, request_id: str) -> RequestCollector | None:
        with self._lock:
            return self._collectors.get(request_id)
