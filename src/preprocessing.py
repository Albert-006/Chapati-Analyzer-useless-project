"""
Image Preprocessing Module for Chapati Analyzer™.
Handles image loading, validation, scaling, color space conversions, and noise filtering.
"""

from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np
from PIL import Image
import io


class ImageValidationError(ValueError):
    """Raised when an uploaded/captured image fails validation checks."""
    pass


class Preprocessor:
    """Preprocesses input images for computer vision analysis without destroying originals."""

    MAX_DIMENSION = 1200
    MIN_DIMENSION = 100
    MAX_FILE_BYTES = 20 * 1024 * 1024  # 20 MB

    @staticmethod
    def load_and_validate(image_bytes: bytes) -> np.ndarray:
        """
        Loads image bytes, verifies format, dimensions, and readability.
        Returns original image in BGR format (OpenCV convention).
        """
        if not image_bytes:
            raise ImageValidationError("Empty image payload provided.")

        if len(image_bytes) > Preprocessor.MAX_FILE_BYTES:
            raise ImageValidationError(
                f"Image file exceeds maximum limit of {Preprocessor.MAX_FILE_BYTES // (1024 * 1024)}MB."
            )

        # Use PIL to safely inspect header without decoding whole image if corrupted
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
            pil_img.verify()
            format_name = pil_img.format
            if format_name not in ["JPEG", "PNG", "WEBP", "MPO"]:
                raise ImageValidationError(f"Unsupported image format: {format_name}. Expected JPEG, PNG, or WebP.")
        except Exception as e:
            raise ImageValidationError(f"Cannot read image file. Image may be corrupted: {str(e)}")

        # Decode using OpenCV
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise ImageValidationError("Failed to decode image pixels into OpenCV matrix.")

        h, w = img_bgr.shape[:2]
        if h < Preprocessor.MIN_DIMENSION or w < Preprocessor.MIN_DIMENSION:
            raise ImageValidationError(
                f"Image dimensions ({w}x{h}) are too small for geometric analysis. Minimum is {Preprocessor.MIN_DIMENSION}x{Preprocessor.MIN_DIMENSION}px."
            )

        return img_bgr

    @staticmethod
    def preprocess(img_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Performs resizing, color conversions, and noise filtering.
        Returns a dictionary containing working copies and color spaces:
        - original: unmodified BGR
        - working_bgr: resized BGR image for processing
        - scale_factor: ratio (working_dim / original_dim)
        - gray: smoothed grayscale
        - hsv: HSV color space
        - lab: CIE Lab color space
        """
        original = img_bgr.copy()
        h, w = original.shape[:2]

        scale_factor = 1.0
        max_dim = max(h, w)
        if max_dim > Preprocessor.MAX_DIMENSION:
            scale_factor = Preprocessor.MAX_DIMENSION / float(max_dim)
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))
            working_bgr = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            working_bgr = original.copy()

        # Convert color spaces
        gray_raw = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)

        # Gentle bilateral filter to smooth flour speckles while preserving strong edges
        gray_smooth = cv2.bilateralFilter(gray_raw, d=9, sigmaColor=75, sigmaSpace=75)

        hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2LAB)

        return {
            "original_bgr": original,
            "working_bgr": working_bgr,
            "scale_factor": scale_factor,
            "gray_raw": gray_raw,
            "gray_smooth": gray_smooth,
            "hsv": hsv,
            "lab": lab,
            "height": working_bgr.shape[0],
            "width": working_bgr.shape[1],
        }
