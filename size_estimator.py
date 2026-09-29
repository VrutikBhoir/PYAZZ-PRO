"""Pixel -> mm conversion for a FIXED camera (calibrated once, no per-photo reference)."""
from __future__ import annotations
import math

def onion_diameter_mm(onion_px_diameter: float, pixels_per_mm: float) -> float:
    if pixels_per_mm <= 0:
        raise ValueError("pixels_per_mm must be positive (run calibrate.py first)")
    if onion_px_diameter <= 0:
        raise ValueError("onion pixel diameter must be positive")
    return onion_px_diameter / pixels_per_mm

def px_diameter_from_bbox(w_px: float, h_px: float) -> float:
    """Mean diameter from bbox (robust to slight ellipse)."""
    return (float(w_px) + float(h_px)) / 2.0

def pixels_per_mm_from_reference(pixel_length: float, real_length_mm: float) -> float:
    """Calculate fixed-camera calibration from a measured ruler segment."""
    if pixel_length <= 0:
        raise ValueError("pixel_length must be positive")
    if real_length_mm <= 0:
        raise ValueError("real_length_mm must be positive")
    return float(pixel_length) / float(real_length_mm)

def px_diameter_from_mask_area(mask_area_px: float) -> float:
    """Equivalent-circle diameter from segmentation mask area: d = 2*sqrt(A/pi)."""
    if mask_area_px <= 0:
        raise ValueError("mask area must be positive")
    return 2.0 * math.sqrt(mask_area_px / math.pi)
