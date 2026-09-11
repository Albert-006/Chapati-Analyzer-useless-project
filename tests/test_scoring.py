"""
Unit tests for scoring, verdicts, and deterministic calculations.
Verifies:
- Geometry-first weights
- Strict determinism (no random values)
- Verdict and quote mappings for all tiers
- Samosa / Rounded triangle prevention from Grandma approval
"""

import unittest
from src.models import (
    GeometricMetrics,
    RadialMetrics,
    BrowningMetrics,
    TextureMetrics,
)
from src.scoring import Scorer


class TestScoring(unittest.TestCase):

    def _create_mock_metrics(self, circularity=0.95, stability=95.0, symmetry=95.0,
                             smoothness=95.0, center_acc=95.0, browning_score=95.0,
                             texture_score=95.0, ideal_dev=95.0, corner_score=95.0,
                             ellipse_round=95.0, geom_type="NEAR CIRCLE"):
        geom = GeometricMetrics(
            area=50000.0, perimeter=800.0, circularity=circularity,
            circularity_raw=circularity, circularity_light=circularity,
            circularity_moderate=circularity,
            bounding_box=(50, 50, 250, 250), width=250.0, height=250.0,
            aspect_ratio=1.0, equivalent_diameter=252.0, solidity=0.99,
            ellipse_major_axis=252.0, ellipse_minor_axis=250.0,
            ellipse_axis_ratio=1.008, ellipse_eccentricity=0.12,
            ellipse_roundness_score=ellipse_round, centroid=(175.0, 175.0),
            bbox_center=(175.0, 175.0), fitted_circle_center=(175.0, 175.0),
            fitted_circle_radius=126.0, center_deviation_px=0.0,
            center_deviation_normalized=0.0, center_accuracy_score=center_acc
        )
        radial = RadialMetrics(
            angles_deg=list(range(360)), radii=[126.0]*360, mean_radius=126.0,
            min_radius=126.0, max_radius=126.0, std_radius=0.5,
            coefficient_of_variation=0.004, range_over_mean=0.01,
            median_absolute_deviation=0.3, max_radial_deviation_normalized=0.005,
            radius_stability_score=stability, symmetry_diff_180=0.0,
            symmetry_score=symmetry, boundary_roughness=0.001,
            boundary_smoothness_score=smoothness,
            ideal_circle_mean_error=0.005, ideal_circle_rms_error=0.008,
            ideal_circle_max_error=0.01, ideal_circle_p95_error=0.009,
            ideal_circle_deviation_score=ideal_dev,
            k2_ellipse=0.5, k3_triangle=0.4, k4_lobed=0.3,
            harmonic_irregularity=1.2, harmonic_score=96.0,
            corner_count=0, corner_strength="LOW", max_turning_angle=45.0,
            corner_quality_score=corner_score, geometric_type=geom_type
        )
        browning = BrowningMetrics(
            chapati_area_px=50000, browned_area_px=2500, burn_ratio=0.05,
            patch_count=8, largest_patch_px=200, mean_browning_intensity=120.0,
            browning_control_score=browning_score, classification="Balanced Browning"
        )
        texture = TextureMetrics(
            grayscale_std=25.0, laplacian_variance=120.0, local_contrast=15.0,
            texture_score=texture_score
        )
        return geom, radial, browning, texture

    def test_deterministic_scoring(self):
        geom, radial, browning, texture = self._create_mock_metrics()
        score1 = Scorer.calculate_perfection(geom, radial, browning, texture)
        score2 = Scorer.calculate_perfection(geom, radial, browning, texture)

        self.assertEqual(score1.perfection_index, score2.perfection_index)
        self.assertEqual(score1.verdict, score2.verdict)

    def test_verdict_thresholds(self):
        # 95-100: GEOMETRICALLY ENLIGHTENED
        v, q = Scorer.get_verdict(97.5)
        self.assertEqual(v, "GEOMETRICALLY ENLIGHTENED")
        self.assertIn("mathematicians", q)

        # 90-95: ALMOST CIRCULAR
        v, q = Scorer.get_verdict(92.0)
        self.assertEqual(v, "ALMOST CIRCULAR")
        self.assertIn("Grandma", q)

        # 80-90: ACCEPTABLE CHAPATI
        v, q = Scorer.get_verdict(85.4)
        self.assertEqual(v, "ACCEPTABLE CHAPATI")
        self.assertIn("geographically confused", q)

        # 70-80: QUESTIONABLE GEOMETRY
        v, q = Scorer.get_verdict(74.0)
        self.assertEqual(v, "QUESTIONABLE GEOMETRY")
        self.assertIn("suggestion", q)

        # 60-70: SHAPE INCIDENT
        v, q = Scorer.get_verdict(65.0)
        self.assertEqual(v, "SHAPE INCIDENT")
        self.assertIn("Something happened here", q)

        # Below 60: PLEASE CONSULT A CHAPATI ENGINEER
        v, q = Scorer.get_verdict(52.0)
        self.assertEqual(v, "PLEASE CONSULT A CHAPATI ENGINEER")
        self.assertIn("intervention", q)

    def test_rounded_triangle_cannot_get_grandma_approval(self):
        """Even with high circularity, a rounded triangle must never get ALMOST CIRCULAR."""
        v, q = Scorer.get_verdict(88.0, geometric_type="ROUNDED TRIANGULAR")
        self.assertEqual(v, "QUESTIONABLE GEOMETRY")
        self.assertIn("triangular", q.lower())


if __name__ == "__main__":
    unittest.main()
