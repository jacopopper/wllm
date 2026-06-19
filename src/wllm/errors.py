from __future__ import annotations


class WLLMError(RuntimeError):
    """Base exception for package-level operational errors."""


class DependencyUnavailable(WLLMError):
    """Raised when an optional runtime dependency is required but missing."""


class TraceUnavailable(WLLMError):
    """Raised when a requested trace feature cannot be collected."""

