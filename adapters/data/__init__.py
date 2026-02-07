"""Data adapters."""

from .data_loader import DataLoader

# Keep old name for backward compatibility
CSVDataLoader = DataLoader

__all__ = ["DataLoader", "CSVDataLoader"]


