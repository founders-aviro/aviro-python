"""Tests for aviro package."""

import pytest
import aviro


def test_version():
    """Test that version is defined."""
    assert hasattr(aviro, "__version__")
    assert aviro.__version__ is not None


def test_import():
    """Test that package can be imported."""
    assert aviro is not None
