"""
Visual Texture Analysis Module for Chapati Analyzer™.
Calculates visible surface texture, Laplacian variance, and micro-structure
within the segmented chapati mask.
"""

from typing import Tuple
import cv2
import numpy as np
from src.models import TextureMetrics


class TextureAnalyzer:
    """Extracts optical texture characteristics from the chapati surface."""

    @classmethod
    def analyze(cls, working_bgr: np.ndarray, chapati_mask: np.ndarray) -> TextureMetrics:
        """
        Computes visible texture statistics on the pixels strictly inside the chapati boundary.
        """
        gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
        chapati_pixels = gray[chapati_mask > 0]

        if len(chapati_pixels) == 0:
            raise ValueError("Empty chapati mask provided for texture analysis.")

        # 1. Grayscale standard deviation (global variation)
        gray_std = float(np.std(chapati_pixels))

        # 2. Laplacian variance (micro-texture / sharpness)
        # Compute Laplacian on the grayscale image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Only take variance inside the eroded mask (avoid boundary edge discontinuity)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        eroded_mask = cv2.erode(chapati_mask, kernel)

        interior_pixels = laplacian[eroded_mask > 0]
        if len(interior_pixels) > 0:
            laplacian_var = float(np.var(interior_pixels))
        else:
            laplacian_var = float(np.var(laplacian[chapati_mask > 0]))

        # 3. Local contrast (standard deviation of small patches)
        blur = cv2.blur(gray.astype(np.float64), (7, 7))
        sq_blur = cv2.blur(gray.astype(np.float64)**2, (7, 7))
        local_var = np.maximum(0.0, sq_blur - blur**2)
        local_std = np.sqrt(local_var)
        local_contrast = float(np.mean(local_std[chapati_mask > 0]))

        # Texture Score (0 - 100):
        # A good chapati has moderate, even dough texture (neither completely featureless blur nor noisy granite)
        # Typical laplacian variance ranges from 40 to 400.
        # Grayscale std ranges from 15 to 45.
        score_std = min(100.0, max(20.0, 100.0 - abs(gray_std - 28.0) * 2.2))
        score_lap = min(100.0, max(20.0, 100.0 - abs(np.log1p(laplacian_var) - 5.2) * 18.0))

        texture_score = float(np.clip(0.6 * score_std + 0.4 * score_lap, 10.0, 100.0))

        return TextureMetrics(
            grayscale_std=gray_std,
            laplacian_variance=laplacian_var,
            local_contrast=local_contrast,
            texture_score=texture_score,
        )
