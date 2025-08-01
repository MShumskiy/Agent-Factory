"""
Agent Factory - AI Agent Development Platform

This package provides production-ready AI agents and data processing pipelines
with integrated experimentation capabilities.
"""

__version__ = "0.1.0"
__author__ = "MS"

# Import main components for easy access (with error handling for optional dependencies)
try:
    from . import agents
except ImportError as e:
    print(f"Warning: Could not import agents module: {e}")
    agents = None

try:
    from . import pipelines
except ImportError as e:
    print(f"Warning: Could not import pipelines module: {e}")
    pipelines = None

try:
    from . import core
except ImportError as e:
    print(f"Warning: Could not import core module: {e}")
    core = None

__all__ = [
    "agents",
    "pipelines", 
    "core",
    "api",
    "config"
]
