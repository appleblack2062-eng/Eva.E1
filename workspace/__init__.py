"""Workspace module for spatial awareness and file system graph management."""

from .filesystem_graph import FileSystemGraph, FileNode
from .code_analyzer import CodeAnalyzer
from .spatial_context import SpatialContextGenerator

__all__ = [
    "FileSystemGraph",
    "FileNode",
    "CodeAnalyzer",
    "SpatialContextGenerator",
]
