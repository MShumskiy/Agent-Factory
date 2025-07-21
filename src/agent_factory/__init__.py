"""
Agent Factory - AI Agent Development Platform

This package provides production-ready AI agents and data processing pipelines
with integrated experimentation capabilities.
"""

__version__ = "0.1.0"
__author__ = "MS"

# Import main components for easy access
from .agents import *
from .pipelines import *
from .core import *

__all__ = [
    "agents",
    "pipelines", 
    "core",
    "api",
    "config"
]
