"""
Browning Analysis Module for Chapati Analyzer™.
Extracts Maillard reaction browning patches, computes burn ratio,
analyzes 5 concentric radial zones, and produces humorous pattern classifications.
"""

from typing import Dict, Any, List, Tuple
import cv2
import numpy as np
from src.models import BrowningMetrics, BrowningZone


class BrowningAnalyzer:
    """Analyzes toasting, browning spots, and radial burn distribution."""

    ZONE_DEFINITIONS = [
        ("Center", 0.00, 0.20),
        ("Inner", 0.20, 0.40),
        ("Middle", 0.40, 0.60),
        ("Outer", 0.60, 0.80),
        ("Edge", 0.80, 1.05),  # allow slight margin for contour wobble
    ]

    @classmethod
    def analyze(
        cls,
        working_bgr: np.ndarray,
        chapati_mask: np.ndarray,
        centroid: Tuple[float, float],
        equivalent_radius: float,
    ) -> Tuple[BrowningMetrics, np.ndarray]:
        """
        Analyzes the browning patches within the chapati mask.
        Returns BrowningMetrics and a binary browning_mask (255 for browned pixels).
        """
        chapati_area_px = int(np.count_nonzero(chapati_mask))
        if chapati_area_px == 0:
            raise ValueError("Chapati mask has zero pixels.")

        # Convert to HSV and Grayscale
        hsv = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(working_bgr, cv2.COLOR_BGR2GRAY)
        v_channel = hsv[:, :, 2]  # Value / Brightness

        # Compute baseline dough brightness inside the chapati
        dough_pixels_v = v_channel[chapati_mask > 0]
        dough_median_v = float(np.median(dough_pixels_v))

        # Browning patches are significantly darker than dough median
        # and have toasting coloration (lower value, or high local contrast)
        # We also use adaptive local contrast to avoid mistaking global gradients for burnt patches
        blurred_v = cv2.GaussianBlur(v_channel, (31, 31), 0)
        local_drop = blurred_v.astype(np.float32) - v_channel.astype(np.float32)

        # A pixel is considered browned if it is:
        # 1. Inside the chapati mask
        # 2. Has noticeable local drop (> 22) OR is deeply dark compared to dough median (v < 0.60 * median)
        is_dark_spot = (v_channel < (0.68 * dough_median_v))
        is_local_char = (local_drop > 24)

        browning_candidates = (is_dark_spot | is_local_char) & (chapati_mask > 0)
        browning_mask_raw = np.uint8(browning_candidates) * 255

        # Morphological opening to remove tiny noise/grain
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        browning_mask = cv2.morphologyEx(browning_mask_raw, cv2.MORPH_OPEN, kernel)

        browned_area_px = int(np.count_nonzero(browning_mask))
        burn_ratio = float(browned_area_px) / float(chapati_area_px)

        # Find individual patches
        patch_contours, _ = cv2.findContours(browning_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        patch_count = len(patch_contours)
        largest_patch_px = int(max([cv2.contourArea(c) for c in patch_contours], default=0))

        # Mean intensity of browning
        if browned_area_px > 0:
            mean_intensity = float(np.mean(255 - gray[browning_mask > 0]))  # higher = darker burn
        else:
            mean_intensity = 0.0

        # Radial Zones Analysis
        cx, cy = centroid
        h, w = chapati_mask.shape
        y_grid, x_grid = np.ogrid[:h, :w]
        dist_from_center = np.hypot(x_grid - cx, y_grid - cy)

        zones: List[BrowningZone] = []
        zone_densities: Dict[str, float] = {}

        for name, r_min_frac, r_max_frac in cls.ZONE_DEFINITIONS:
            r_min_px = r_min_frac * equivalent_radius
            r_max_px = r_max_frac * equivalent_radius

            zone_pixel_mask = (
                (dist_from_center >= r_min_px) &
                (dist_from_center < r_max_px) &
                (chapati_mask > 0)
            )
            zone_area = int(np.count_nonzero(zone_pixel_mask))
            if zone_area > 0:
                zone_browned = int(np.count_nonzero(browning_mask & zone_pixel_mask))
                density = float(zone_browned) / float(zone_area)
                zone_mean_int = float(np.mean(255 - gray[browning_mask & zone_pixel_mask])) if zone_browned > 0 else 0.0
            else:
                zone_browned = 0
                density = 0.0
                zone_mean_int = 0.0

            zone = BrowningZone(
                name=name,
                radius_fraction_min=r_min_frac,
                radius_fraction_max=r_max_frac,
                area_px=zone_area,
                browned_pixels=zone_browned,
                browning_density=density,
                mean_intensity=zone_mean_int,
            )
            zones.append(zone)
            zone_densities[name] = density

        # Classification heuristics
        classification = cls._classify_pattern(burn_ratio, zone_densities)

        # Browning Control Score (0 - 100)
        # Optimal chapati has 3% - 10% browning: Score 90-100
        # Under-cooked (< 1.5%): Score 70-80
        # Charred (> 20%): Score drops sharply
        browning_control_score = cls._calculate_browning_score(burn_ratio)

        metrics = BrowningMetrics(
            chapati_area_px=chapati_area_px,
            browned_area_px=browned_area_px,
            burn_ratio=burn_ratio,
            patch_count=patch_count,
            largest_patch_px=largest_patch_px,
            mean_browning_intensity=mean_intensity,
            browning_control_score=browning_control_score,
            classification=classification,
            zones=zones,
        )

        return metrics, browning_mask

    @classmethod
    def _classify_pattern(cls, burn_ratio: float, zone_densities: Dict[str, float]) -> str:
        """Humorous classification of toasting pattern based on burn ratio and zone density."""
        if burn_ratio < 0.008:
            return "Minimal Visible Browning"
        if burn_ratio > 0.25:
            return "Extremely Charred"
        if burn_ratio > 0.15:
            return "Mildly Charred"

        center_burn = zone_densities.get("Center", 0.0) + zone_densities.get("Inner", 0.0)
        edge_burn = zone_densities.get("Outer", 0.0) + zone_densities.get("Edge", 0.0)

        if center_burn > 2.2 * (edge_burn + 1e-4) and center_burn > 0.06:
            return "Center Burner"
        if edge_burn > 2.2 * (center_burn + 1e-4) and edge_burn > 0.06:
            return "Edge Burner"
        if 0.03 <= burn_ratio <= 0.14:
            return "Balanced Browning"
        return "Random Browning"

    @classmethod
    def _calculate_browning_score(cls, burn_ratio: float) -> float:
        """Scores browning control: peak score around 5% - 8% toasting."""
        if burn_ratio < 0.005:
            return 72.0  # Slightly under-toasted / pale
        elif burn_ratio <= 0.08:
            # Ideal toasting curve: 75 -> 100
            return 75.0 + (burn_ratio / 0.08) * 25.0
        elif burn_ratio <= 0.15:
            # Tolerable toasting: 100 -> 80
            return 100.0 - ((burn_ratio - 0.08) / 0.07) * 20.0
        elif burn_ratio <= 0.28:
            # Charred: 80 -> 40
            return 80.0 - ((burn_ratio - 0.15) / 0.13) * 40.0
        else:
            # Severely scorched
            return max(5.0, 40.0 - (burn_ratio - 0.28) * 100.0)
