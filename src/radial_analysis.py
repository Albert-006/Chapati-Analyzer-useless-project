"""
Radial, Symmetry, and Edge Analysis Module for Chapati Analyzer™.
Samples r(theta) across 360 degrees, computes coefficient of variation,
ideal-circle residual analysis (RMS/P95), Fourier harmonic shape analysis (FFT),
polygon corner/turning-angle curvature analysis, and deterministic geometric classification.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter1d
from src.models import RadialMetrics


class RadialAnalyzer:
    """Computes radial metrics, ideal-circle deviation, Fourier harmonics, and corner curvature."""

    NUM_SAMPLES = 360

    @classmethod
    def sample_radial_profile(
        cls, contour: np.ndarray, centroid: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Converts the contour into a clean 360-element radial signal r(theta),
        where theta is in degrees [0, 359].
        """
        cx, cy = centroid
        pts = contour.reshape(-1, 2).astype(np.float64)

        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy

        # Angle in degrees [0, 360)
        angles_deg = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        radii = np.hypot(dx, dy)

        # Sort by angle
        sort_idx = np.argsort(angles_deg)
        angles_sorted = angles_deg[sort_idx]
        radii_sorted = radii[sort_idx]

        # For periodic interpolation across 0/360 wrap-around
        angles_extended = np.concatenate([
            angles_sorted - 360.0,
            angles_sorted,
            angles_sorted + 360.0
        ])
        radii_extended = np.concatenate([radii_sorted, radii_sorted, radii_sorted])

        # Target sample grid: 0, 1, 2, ..., 359
        target_angles = np.arange(cls.NUM_SAMPLES, dtype=np.float64)
        sampled_radii = np.interp(target_angles, angles_extended, radii_extended)

        return target_angles, sampled_radii

    @classmethod
    def analyze(
        cls,
        contour: np.ndarray,
        centroid: Tuple[float, float],
        fitted_circle_radius: float,
        ellipse_axis_ratio: float = 1.0,
        circularity: float = 1.0,
    ) -> RadialMetrics:
        """
        Comprehensive radial analysis including ideal circle deviation,
        Fourier harmonic decomposition, and corner turning angles.
        """
        target_angles, radii = cls.sample_radial_profile(contour, centroid)

        mean_r = float(np.mean(radii))
        min_r = float(np.min(radii))
        max_r = float(np.max(radii))
        std_r = float(np.std(radii))

        # 1. Statistics & Stability
        cv = std_r / mean_r if mean_r > 0 else 1.0
        range_over_mean = (max_r - min_r) / mean_r if mean_r > 0 else 1.0
        mad = float(np.median(np.abs(radii - np.median(radii))))
        max_rad_dev = float(np.max(np.abs(radii - mean_r)) / mean_r) if mean_r > 0 else 1.0

        # Radius stability score: penalizes high coefficient of variation
        radius_stability_score = max(0.0, min(100.0, 100.0 * (1.0 - 2.6 * cv)))

        # 2. Rotational Symmetry
        radii_180 = np.roll(radii, -180)
        diff_180 = np.mean(np.abs(radii - radii_180)) / mean_r if mean_r > 0 else 1.0
        radii_90 = np.roll(radii, -90)
        diff_90 = np.mean(np.abs(radii - radii_90)) / mean_r if mean_r > 0 else 1.0

        weighted_diff = 0.70 * diff_180 + 0.30 * diff_90
        symmetry_score = max(0.0, min(100.0, 100.0 * (1.0 - 2.2 * weighted_diff)))

        # 3. Ideal Circle Residual Analysis
        r_ideal = fitted_circle_radius
        norm_errors = np.abs(radii - r_ideal) / mean_r if mean_r > 0 else np.zeros_like(radii)
        mean_err = float(np.mean(norm_errors))
        rms_err = float(np.sqrt(np.mean(norm_errors**2)))
        max_err = float(np.max(norm_errors))
        p95_err = float(np.percentile(norm_errors, 95))

        # Ideal circle deviation score: 100 for zero deviation, drops sharply with RMS error
        # RMS 1.5% -> ~93.2, RMS 5.0% -> ~77.5, RMS 7.2% -> ~67.6, RMS 12% -> ~46.0
        ideal_circle_deviation_score = max(0.0, min(100.0, 100.0 * (1.0 - 4.5 * rms_err)))

        # 4. Fourier Harmonic Decomposition (FFT)
        zero_centered = radii - mean_r
        fft_complex = np.fft.rfft(zero_centered)
        fft_magnitudes = np.abs(fft_complex)

        # Harmonic amplitudes as percentage of mean radius: A_k = (2 * |X[k]| / N) / mean_r * 100
        N = float(cls.NUM_SAMPLES)
        amp_pct = (2.0 * fft_magnitudes / (N * mean_r)) * 100.0 if mean_r > 0 else np.zeros_like(fft_magnitudes)

        k2 = float(amp_pct[2]) if len(amp_pct) > 2 else 0.0  # Elliptical (2-lobed)
        k3 = float(amp_pct[3]) if len(amp_pct) > 3 else 0.0  # Triangular (3-lobed / samosa)
        k4 = float(amp_pct[4]) if len(amp_pct) > 4 else 0.0  # Quadrilateral (4-lobed)

        # Sum of non-circular harmonic amplitudes (k=2 through k=6) in percentage points
        harmonic_irregularity = float(np.sum(amp_pct[2:7])) if len(amp_pct) > 6 else 0.0
        harmonic_score = max(0.0, min(100.0, 100.0 - (harmonic_irregularity * 3.0)))

        # 5. Corner / Vertex Curvature Analysis
        peri = cv2.arcLength(contour, closed=True)
        # Approximate with 0.02 * perimeter to evaluate vertex angular uniformity
        poly = cv2.approxPolyDP(contour, 0.02 * peri, closed=True)
        pts_poly = poly.reshape(-1, 2)
        N_pts = len(pts_poly)

        turning_angles = []
        sharp_corners = 0
        max_turn = 0.0

        if N_pts >= 3:
            for i in range(N_pts):
                v1 = pts_poly[i] - pts_poly[(i - 1) % N_pts]
                v2 = pts_poly[(i + 1) % N_pts] - pts_poly[i]
                norm_v1 = np.linalg.norm(v1)
                norm_v2 = np.linalg.norm(v2)
                if norm_v1 > 0 and norm_v2 > 0:
                    cos_val = np.dot(v1, v2) / (norm_v1 * norm_v2)
                    turn = float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))
                    turning_angles.append(turn)
                    if turn > 65.0:
                        sharp_corners += 1
                    max_turn = max(max_turn, turn)

        turn_std = float(np.std(turning_angles)) if turning_angles else 0.0

        # Classify corner strength using joint harmonic k3 and turning angle variance
        # A circle has turn_std ~ 0-5. A triangle has turn_std > 14 and k3 > 6.0%.
        if (k3 >= 6.0 and turn_std >= 13.0) or (sharp_corners >= 2 and k3 >= 5.0):
            corner_strength = "HIGH"
            corner_quality_score = max(35.0, 100.0 - 30.0 - (k3 * 4.0))
        elif (k3 >= 3.5 and turn_std >= 10.0) or sharp_corners >= 1:
            corner_strength = "MODERATE"
            corner_quality_score = max(60.0, 100.0 - 15.0 - (k3 * 2.5))
        else:
            corner_strength = "LOW"
            corner_quality_score = max(82.0, min(100.0, 100.0 - (k3 * 2.0)))

        # 6. Boundary Smoothness (decoupled from circularity)
        padded_radii = np.pad(radii, 20, mode="wrap")
        smoothed_padded = gaussian_filter1d(padded_radii, sigma=3.0)
        smoothed = smoothed_padded[20:-20]
        residuals = radii - smoothed
        residual_std = float(np.std(residuals))
        normalized_roughness = residual_std / mean_r if mean_r > 0 else 1.0
        boundary_smoothness_score = max(0.0, min(100.0, 100.0 * (1.0 - 15.0 * normalized_roughness)))

        # 7. Deterministic Geometric Type Classification
        geometric_type = cls._classify_geometric_type(
            circularity=circularity,
            rms_error=rms_err,
            axis_ratio=ellipse_axis_ratio,
            k2=k2,
            k3=k3,
            k4=k4,
            corner_strength=corner_strength,
            harmonic_irregularity=harmonic_irregularity,
        )

        return RadialMetrics(
            angles_deg=target_angles.tolist(),
            radii=radii.tolist(),
            mean_radius=mean_r,
            min_radius=min_r,
            max_radius=max_r,
            std_radius=std_r,
            coefficient_of_variation=float(cv),
            range_over_mean=float(range_over_mean),
            median_absolute_deviation=mad,
            max_radial_deviation_normalized=max_rad_dev,
            radius_stability_score=radius_stability_score,
            symmetry_diff_180=float(diff_180),
            symmetry_score=symmetry_score,
            boundary_roughness=normalized_roughness,
            boundary_smoothness_score=boundary_smoothness_score,
            ideal_circle_mean_error=mean_err,
            ideal_circle_rms_error=rms_err,
            ideal_circle_max_error=max_err,
            ideal_circle_p95_error=p95_err,
            ideal_circle_deviation_score=ideal_circle_deviation_score,
            k2_ellipse=k2,
            k3_triangle=k3,
            k4_lobed=k4,
            harmonic_irregularity=harmonic_irregularity,
            harmonic_score=harmonic_score,
            corner_count=sharp_corners,
            corner_strength=corner_strength,
            max_turning_angle=max_turn,
            corner_quality_score=corner_quality_score,
            geometric_type=geometric_type,
        )

    @staticmethod
    def _classify_geometric_type(
        circularity: float,
        rms_error: float,
        axis_ratio: float,
        k2: float,
        k3: float,
        k4: float,
        corner_strength: str,
        harmonic_irregularity: float,
    ) -> str:
        """Assigns deterministic geometric classification."""
        if circularity >= 0.97 and rms_error < 0.024 and axis_ratio < 1.03 and corner_strength == "LOW":
            return "PERFECT CIRCLE"
        if circularity >= 0.93 and rms_error < 0.038 and axis_ratio < 1.06 and corner_strength == "LOW":
            return "NEAR CIRCLE"
        if corner_strength == "HIGH" or (k3 >= 6.0 and k3 > 1.3 * k2):
            return "ROUNDED TRIANGULAR"
        if axis_ratio >= 1.14 and k2 > 1.4 * k3:
            return "OVAL"
        if k4 >= 5.5 or (harmonic_irregularity > 12.0 and k4 > k3):
            return "MULTI-LOBED"
        if rms_error > 0.09 or circularity < 0.78:
            return "HIGHLY IRREGULAR"
        return "ROUND / IRREGULAR"
