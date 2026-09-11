"""
Unit tests for radial analysis, symmetry, and edge smoothness.
Verifies:
- Radius variation & stability score
- Rotational symmetry (180° / 90°)
- Ideal circle residual & Fourier harmonics
- Boundary smoothness score
"""

import unittest
import numpy as np
from tests.synthetic_shapes import SyntheticShapeGenerator
from src.geometric_analysis import GeometricAnalyzer
from src.radial_analysis import RadialAnalyzer


class TestRadial(unittest.TestCase):

    def test_perfect_circle_radial_stability(self):
        _, contour = SyntheticShapeGenerator.create_perfect_circle(radius=150)
        geom = GeometricAnalyzer.analyze(contour)
        radial = RadialAnalyzer.analyze(
            contour, geom.centroid, geom.fitted_circle_radius,
            ellipse_axis_ratio=geom.ellipse_axis_ratio,
            circularity=geom.circularity
        )

        # 360 angular samples
        self.assertEqual(len(radial.angles_deg), 360)
        self.assertEqual(len(radial.radii), 360)

        # Radius variation in a raster circle should be near 0
        self.assertLess(radial.coefficient_of_variation, 0.02)
        self.assertGreaterEqual(radial.radius_stability_score, 95.0)

        # Perfect circle has near 100 symmetry and smoothness
        self.assertGreaterEqual(radial.symmetry_score, 95.0)
        self.assertGreaterEqual(radial.boundary_smoothness_score, 95.0)

        # Ideal circle residual should be very low
        self.assertLess(radial.ideal_circle_rms_error, 0.02)
        self.assertGreaterEqual(radial.ideal_circle_deviation_score, 90.0)

    def test_stability_monotonicity(self):
        _, circle_cnt = SyntheticShapeGenerator.create_perfect_circle(radius=140)
        _, irregular_cnt = SyntheticShapeGenerator.create_slightly_irregular_circle(radius=140, wobble=0.08)
        _, polygon_cnt = SyntheticShapeGenerator.create_irregular_polygon()

        c_geom = GeometricAnalyzer.analyze(circle_cnt)
        i_geom = GeometricAnalyzer.analyze(irregular_cnt)
        p_geom = GeometricAnalyzer.analyze(polygon_cnt)

        c_radial = RadialAnalyzer.analyze(circle_cnt, c_geom.centroid, c_geom.fitted_circle_radius)
        i_radial = RadialAnalyzer.analyze(irregular_cnt, i_geom.centroid, i_geom.fitted_circle_radius)
        p_radial = RadialAnalyzer.analyze(polygon_cnt, p_geom.centroid, p_geom.fitted_circle_radius)

        # Stability: Circle > Irregular Circle > Polygon
        self.assertGreater(c_radial.radius_stability_score, i_radial.radius_stability_score)
        self.assertGreater(i_radial.radius_stability_score, p_radial.radius_stability_score)

    def test_ellipse_symmetry(self):
        """An ellipse has strong 180-degree rotational symmetry."""
        _, ellipse_cnt = SyntheticShapeGenerator.create_ellipse(axis_a=150, axis_b=110)
        e_geom = GeometricAnalyzer.analyze(ellipse_cnt)
        e_radial = RadialAnalyzer.analyze(ellipse_cnt, e_geom.centroid, e_geom.fitted_circle_radius)

        # Ellipse has 180° symmetry, so diff_180 should be quite small (< 0.05)
        self.assertLess(e_radial.symmetry_diff_180, 0.05)


if __name__ == "__main__":
    unittest.main()
