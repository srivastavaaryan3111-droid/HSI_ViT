"""Patch extraction entry point; array logic lives in preprocessing.py."""
from preprocessing import PATCH_SIZE, extract_patches

__all__ = ["PATCH_SIZE", "extract_patches"]
