"""Exceptions used across ChurnVec."""

from __future__ import annotations


class ChurnVecError(Exception):
    """Base class for all project-specific errors."""


class EmptyIndexError(ChurnVecError):
    """Raised when a search is attempted on an empty index."""


class DuplicateIDError(ChurnVecError):
    """Raised when an external vector ID is inserted twice."""


class DimensionMismatchError(ChurnVecError, ValueError):
    """Raised when a vector has the wrong dimensionality."""


class InvariantError(ChurnVecError, AssertionError):
    """Raised when the graph fails an internal consistency check."""
