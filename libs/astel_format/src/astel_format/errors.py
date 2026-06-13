"""Exception types for astel_format."""

from __future__ import annotations


class AstelFormatError(Exception):
    """Base class for all astel_format errors."""


class AstelValidationError(AstelFormatError):
    """Manifest failed JSON-Schema or cross-reference validation."""


class PathSecurityError(AstelFormatError):
    """A path was absolute, escaped the package root, or otherwise unsafe."""
