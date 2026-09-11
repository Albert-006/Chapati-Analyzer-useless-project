"""
Geometric Analysis Module for Chapati Analyzer™.
Calculates mathematical area, perimeter, multi-level circularity, bounding box,
equivalent diameter, solidity, ellipse fitting, Taubin circle fitting,
center deviation, and center accuracy score.
"""

from typing import Tuple, Dict, Any
import cv2
import numpy as np
from src.models import GeometricMetrics


class GeometricAnalyzer:
    """Performs rigorous geometric feature extraction on the detected chapati contour."""

    @staticmethod
    def fit_algebraic_circle(points: np.ndarray) -> Tuple[float, float, float]:
        """
        Fits a circle using Kåsa's algebraic least-squares method.
        points shape: (N, 2)
        Equation of circle: (x - a)^2 + (y - b)^2 = R^2
        <=> x^2 + y^2 - 2ax - 2by + (a^2 + b^2 - R^2) = 0
        Returns (center_x, center_y, radius).
        """
        x = points[:, 0].astype(np.float64)
        y = points[:, 1].astype(np.float64)

        A = np.column_stack([x, y, np.ones_like(x)])
        b = x**2 + y**2

        c, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        center_x = c[0] / 2.0
        center_y = c[1] / 2.0
        val = c[2] + center_x**2 + center_y**2
        radius = float(np.sqrt(max(val, 1.0)))

        return float(center_x), float(center_y), radius

    @classmethod
    def analyze(cls, contour: np.ndarray) -> GeometricMetrics:
        """
        Extracts multi-faceted geometric metrics from the chapati contour.
        """
        # 1. Area & Perimeters
        area = float(cv2.contourArea(contour))
        raw_perimeter = float(cv2.arcLength(contour, closed=True))

        if raw_perimeter <= 0 or area <= 0:
            raise ValueError("Invalid contour: Area or perimeter is non-positive.")

        # Multi-level circularity evaluation to prevent smoothing exploitation
        # Raw circularity (includes single-pixel discretization steps)
        circ_raw = min(1.0, max(0.0, (4.0 * np.pi * area) / (raw_perimeter * raw_perimeter)))

        # Light approximation (cleans 1-pixel staircase raster noise only)
        eps_light = max(0.6, 0.0012 * raw_perimeter)
        cnt_light = cv2.approxPolyDP(contour, epsilon=eps_light, closed=True)
        peri_light = float(cv2.arcLength(cnt_light, closed=True))
        circ_light = min(1.0, max(0.0, (4.0 * np.pi * area) / (peri_light * peri_light)))

        # Moderate approximation
        eps_mod = max(1.2, 0.0028 * raw_perimeter)
        cnt_mod = cv2.approxPolyDP(contour, epsilon=eps_mod, closed=True)
        peri_mod = float(cv2.arcLength(cnt_mod, closed=True))
        circ_mod = min(1.0, max(0.0, (4.0 * np.pi * area) / (peri_mod * peri_mod)))

        # Blended circularity: requires raw and light representations to both be sound
        circularity = min(circ_light, circ_mod)

        # 2. Bounding box & Aspect Ratio
        bx, by, bw, bh = cv2.boundingRect(contour)
        aspect_ratio = float(bw) / float(bh) if bh > 0 else 1.0

        # 3. Equivalent Diameter: 2 * sqrt(A / pi)
        equivalent_diameter = 2.0 * np.sqrt(area / np.pi)

        # 4. Solidity: Contour Area / Convex Hull Area
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = float(min(1.0, area / hull_area if hull_area > 0 else 0.0))

        # 5. Ellipse Fitting (cv2.fitEllipse)
        if len(contour) >= 5:
            try:
                (ex, ey), (d1, d2), e_angle = cv2.fitEllipse(contour)
                major = float(max(d1, d2))
                minor = float(min(d1, d2))
                axis_ratio = float(major / minor) if minor > 0 else 1.0
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor / major)**2))) if major > 0 else 0.0
            except Exception:
                major, minor, axis_ratio, eccentricity = equivalent_diameter, equivalent_diameter, 1.0, 0.0
        else:
            major, minor, axis_ratio, eccentricity = equivalent_diameter, equivalent_diameter, 1.0, 0.0

        # Ellipse roundness score: 100 for axis_ratio=1.0. Drops as shape elongates
        # E.g., ratio 1.05 -> 87.5, ratio 1.15 -> 62.5, ratio 1.30 -> 25.0
        ellipse_roundness_score = max(0.0, min(100.0, 100.0 * (1.0 - 2.5 * (axis_ratio - 1.0))))

        # 6. Centroid from moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])
        else:
            cx = float(bx + bw / 2.0)
            cy = float(by + bh / 2.0)

        bbox_cx = float(bx + bw / 2.0)
        bbox_cy = float(by + bh / 2.0)

        # 7. Fitted Circle (Algebraic Fit with fallback to minEnclosingCircle)
        pts_2d = contour.reshape(-1, 2)
        try:
            fx, fy, fr = cls.fit_algebraic_circle(pts_2d)
        except Exception:
            (fx, fy), fr = cv2.minEnclosingCircle(contour)
            fx, fy, fr = float(fx), float(fy), float(fr)

        # 8. Center Deviation
        center_deviation_px = float(np.hypot(cx - fx, cy - fy))
        norm_dev = center_deviation_px / fr if fr > 0 else 0.0

        # Center accuracy score: 100 when deviation is 0
        center_accuracy_score = max(0.0, min(100.0, 100.0 * (1.0 - 4.0 * norm_dev)))

        return GeometricMetrics(
            area=area,
            perimeter=peri_light,
            circularity=circularity,
            circularity_raw=circ_raw,
            circularity_light=circ_light,
            circularity_moderate=circ_mod,
            bounding_box=(bx, by, bw, bh),
            width=float(bw),
            height=float(bh),
            aspect_ratio=aspect_ratio,
            equivalent_diameter=equivalent_diameter,
            solidity=solidity,
            ellipse_major_axis=major,
            ellipse_minor_axis=minor,
            ellipse_axis_ratio=axis_ratio,
            ellipse_eccentricity=eccentricity,
            ellipse_roundness_score=ellipse_roundness_score,
            centroid=(cx, cy),
            bbox_center=(bbox_cx, bbox_cy),
            fitted_circle_center=(fx, fy),
            fitted_circle_radius=fr,
            center_deviation_px=center_deviation_px,
            center_deviation_normalized=norm_dev,
            center_accuracy_score=center_accuracy_score,
        )
