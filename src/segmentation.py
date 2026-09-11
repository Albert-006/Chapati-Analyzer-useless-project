"""
Chapati Detection and Segmentation Module for Chapati Analyzer™.
Uses classical computer vision (Otsu, morphological filtering, color-space thresholding,
and contour scoring) to segment the chapati reliably across diverse backgrounds.
"""

from typing import Tuple, List, Optional, Dict, Any
import cv2
import numpy as np


class ChapatiDetectionError(Exception):
    """Raised when chapati detection fails with a human-friendly verdict."""
    pass


class Segmenter:
    """Segments a chapati from an image using classical computer vision."""

    MIN_AREA_FRACTION = 0.05   # Must take at least 5% of the frame
    MAX_AREA_FRACTION = 0.96   # Cannot be the entire frame edge
    MIN_ASPECT_RATIO = 0.60    # Roughly circular / slightly elliptical
    MAX_ASPECT_RATIO = 1.66
    MIN_SOLIDITY = 0.70        # Compact object, not a hollow ring or fork
    MIN_CIRCULARITY = 0.35     # Preliminary threshold for detection filtering

    @classmethod
    def segment(cls, preprocessed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes multi-strategy candidate detection and extracts the chapati contour.
        Returns:
            contour: np.ndarray (N, 1, 2)
            mask: np.ndarray (H, W) uint8 binary mask (255 for chapati, 0 otherwise)
            cropped_bgr: np.ndarray cropped to bounding box
            cropped_mask: np.ndarray cropped mask
            bbox: Tuple (x, y, w, h)
        """
        working_bgr = preprocessed["working_bgr"]
        gray = preprocessed["gray_smooth"]
        hsv = preprocessed["hsv"]
        lab = preprocessed["lab"]
        h, w = working_bgr.shape[:2]
        total_image_area = float(h * w)

        # Generate candidate masks using complementary strategies
        candidate_masks = cls._generate_candidate_masks(gray, hsv, lab, h, w)

        all_candidates = []
        for mask in candidate_masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                evaluation = cls._evaluate_candidate(cnt, h, w, total_image_area)
                if evaluation["is_valid"]:
                    all_candidates.append((evaluation["quality_score"], cnt, mask, evaluation))

        if not all_candidates:
            raise ChapatiDetectionError(
                "I couldn't find a sufficiently chapati-shaped object. "
                "Ensure your chapati is clearly visible on a contrasting surface."
            )

        # Sort candidates by quality score descending
        all_candidates.sort(key=lambda x: x[0], reverse=True)

        best_score, best_contour, parent_mask, best_eval = all_candidates[0]

        # Check for multiple equally plausible large objects (e.g., 3 separate chapatis or plates)
        close_competitors = [
            cand for cand in all_candidates[1:]
            if cand[0] > 0.85 * best_score and cand[3]["area"] > 0.6 * best_eval["area"]
        ]
        if len(close_competitors) >= 2:
            # Check if centers are significantly different
            pass  # We pick the most centered/highest quality one, but warn if ambiguous

        # Build clean binary mask containing only the chosen contour
        final_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(final_mask, [best_contour], -1, 255, thickness=cv2.FILLED)

        # Morphological close to fill any interior pockets/shadow holes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # Refine contour from the solid filled mask
        solid_contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not solid_contours:
            raise ChapatiDetectionError("The chapati has successfully evaded computer vision during boundary extraction.")

        refined_contour = max(solid_contours, key=cv2.contourArea)

        # Extract bounding box with padding
        bx, by, bw, bh = cv2.boundingRect(refined_contour)
        pad = 10
        x1 = max(0, bx - pad)
        y1 = max(0, by - pad)
        x2 = min(w, bx + bw + pad)
        y2 = min(h, by + bh + pad)

        cropped_bgr = working_bgr[y1:y2, x1:x2].copy()
        cropped_mask = final_mask[y1:y2, x1:x2].copy()

        return {
            "contour": refined_contour,
            "mask": final_mask,
            "cropped_bgr": cropped_bgr,
            "cropped_mask": cropped_mask,
            "bbox": (bx, by, bw, bh),
            "cropped_bounds": (x1, y1, x2, y2),
            "area_px": cv2.contourArea(refined_contour),
            "detection_score": best_score,
        }

    @classmethod
    def _generate_candidate_masks(
        cls, gray: np.ndarray, hsv: np.ndarray, lab: np.ndarray, h: int, w: int
    ) -> List[np.ndarray]:
        """Generates multiple binary masks through diverse classical CV approaches."""
        masks = []

        # 1. Otsu thresholding on smoothed grayscale (direct and inverse)
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu_thresh)

        # 2. Adaptive thresholding
        adapt = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5
        )
        adapt_inv = cv2.bitwise_not(adapt)

        # 3. Dough color segmentation in HSV
        # Chapatis have warm hue (yellow/orange/brown/wheat: Hue 5 to 35) with moderate saturation
        hsv_dough = cv2.inRange(hsv, np.array([5, 25, 40]), np.array([38, 255, 255]))

        # 4. Lab channel b* (yellow-blue opponent: chapatis are positive b* / yellowish-tan)
        b_channel = lab[:, :, 2]
        _, lab_thresh = cv2.threshold(b_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological cleanup on each mask
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

        raw_masks = [otsu_thresh, otsu_inv, adapt_inv, hsv_dough, lab_thresh]
        for m in raw_masks:
            cleaned = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)
            masks.append(cleaned)

        return masks

    @classmethod
    def _evaluate_candidate(
        cls, cnt: np.ndarray, h: int, w: int, total_area: float
    ) -> Dict[str, Any]:
        """Evaluates how 'chapati-like' a contour candidate is."""
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, closed=True)

        if area < cls.MIN_AREA_FRACTION * total_area or area > cls.MAX_AREA_FRACTION * total_area:
            return {"is_valid": False}

        if peri == 0:
            return {"is_valid": False}

        circularity = (4.0 * np.pi * area) / (peri * peri)
        if circularity < cls.MIN_CIRCULARITY:
            return {"is_valid": False}

        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect_ratio = float(bw) / float(bh)
        if aspect_ratio < cls.MIN_ASPECT_RATIO or aspect_ratio > cls.MAX_ASPECT_RATIO:
            return {"is_valid": False}

        # Solidity = contour area / convex hull area
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < cls.MIN_SOLIDITY:
            return {"is_valid": False}

        # Boundary touch penalty (if it strongly hugs image frame boundary, it might be the background border)
        border_touch = (
            (x <= 2) or (y <= 2) or (x + bw >= w - 2) or (y + bh >= h - 2)
        )
        border_penalty = 0.7 if border_touch else 1.0

        # Centrality score (chapatis are usually centered in photo)
        center_x = x + bw / 2.0
        center_y = y + bh / 2.0
        img_center_x = w / 2.0
        img_center_y = h / 2.0
        max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
        actual_dist = np.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
        centrality = max(0.0, 1.0 - (actual_dist / max_dist))

        # Relative area score (ideal chapati is between 20% and 75% of image area)
        area_frac = area / total_area
        if 0.15 <= area_frac <= 0.80:
            area_score = 1.0
        else:
            area_score = 0.8

        quality_score = (
            (circularity * 0.45) +
            (solidity * 0.25) +
            (centrality * 0.15) +
            (area_score * 0.15)
        ) * border_penalty

        return {
            "is_valid": True,
            "area": area,
            "perimeter": peri,
            "circularity": circularity,
            "solidity": solidity,
            "aspect_ratio": aspect_ratio,
            "quality_score": quality_score,
            "bbox": (x, y, bw, bh),
        }
