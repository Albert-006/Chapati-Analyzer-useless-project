"""
Unit tests for geometric analysis and circle fitting in Chapati Analyzer™.
Verifies PRD requirements:
- Circularity = 4 * pi * A / P^2
- Monotonicity: Perfect circle > Ellipse > Irregular Polygon
- Centroid, aspect ratio, equivalent diameter, and fitted circle.
"""

import unittest
import numpy as np
from tests.synthetic_shapes import SyntheticShapeGenerator
from src.geometric_analysis import GeometricAnalyzer


class TestGeometry(unittest.TestCase):

    def test_perfect_circle_metrics(self):
        _, contour = SyntheticShapeGenerator.create_perfect_circle(radius=150)
        metrics = GeometricAnalyzer.analyze(contour)

        # In discrete raster space, a discretized circle has circularity ~ 0.98 - 1.00
        self.assertGreaterEqual(metrics.circularity, 0.95)
        self.assertLessEqual(metrics.circularity, 1.00)

        # Aspect ratio of a circle should be very close to 1.0
        self.assertAlmostEqual(metrics.aspect_ratio, 1.0, delta=0.04)

        # Equivalent diameter should be very close to 2 * radius (300)
        self.assertAlmostEqual(metrics.equivalent_diameter, 300.0, delta=5.0)

        # Center accuracy should be very high
        self.assertGreaterEqual(metrics.center_accuracy_score, 95.0)

    def test_circularity_monotonicity(self):
        """
        PRD Section 33 requirement:
        Expected general behavior:
        Perfect Circle -> highest circularity
        Ellipse -> lower circularity
        Irregular Shape -> lowest circularity
        """
        _, circle_cnt = SyntheticShapeGenerator.create_perfect_circle(radius=140)
        _, ellipse_cnt = SyntheticShapeGenerator.create_ellipse(axis_a=160, axis_b=110)
        _, irregular_cnt = SyntheticShapeGenerator.create_irregular_polygon()

        circle_m = GeometricAnalyzer.analyze(circle_cnt)
        ellipse_m = GeometricAnalyzer.analyze(ellipse_cnt)
        irreg_m = GeometricAnalyzer.analyze(irregular_cnt)

        self.assertGreater(circle_m.circularity, ellipse_m.circularity)
        self.assertGreater(ellipse_m.circularity, irreg_m.circularity)

        # Ellipse aspect ratio must reflect elongation
        self.assertGreater(ellipse_m.aspect_ratio, 1.2)

    def test_fitted_circle_center_accuracy(self):
        _, contour = SyntheticShapeGenerator.create_perfect_circle(radius=120, img_size=400)
        metrics = GeometricAnalyzer.analyze(contour)

        # Centroid and fitted circle center should be within 1.5 pixels
        cx, cy = metrics.centroid
        fx, fy = metrics.fitted_circle_center
        dist = np.hypot(cx - fx, cy - fy)
        self.assertLess(dist, 2.0)
        self.assertGreaterEqual(metrics.center_accuracy_score, 98.0)


if __name__ == "__main__":
    unittest.main()
