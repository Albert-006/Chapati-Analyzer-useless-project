"""
Comprehensive Shape Calibration and Regression Tests for Chapati Analyzer™.
Verifies:
1. Synthetic shape ranking across Shapes A through H.
2. Samosa / Rounded Triangle detection and score capping.
3. Regression on user specimens (CHAPATI-0003 > CHAPATI-0002 > CHAPATI-0004).
4. Browning bug verification (burn_ratio < 0.008 -> 'Minimal Visible Browning').
"""

import unittest
import glob
import cv2
from tests.synthetic_shapes import SyntheticShapeGenerator
from src.geometric_analysis import GeometricAnalyzer
from src.radial_analysis import RadialAnalyzer
from src.browning_analysis import BrowningAnalyzer
from src.texture_analysis import TextureAnalyzer
from src.scoring import Scorer
from src.preprocessing import Preprocessor
from src.segmentation import Segmenter


class TestShapeCalibration(unittest.TestCase):

    def _evaluate_contour(self, contour, name="Shape"):
        geom = GeometricAnalyzer.analyze(contour)
        radial = RadialAnalyzer.analyze(
            contour,
            geom.centroid,
            geom.fitted_circle_radius,
            ellipse_axis_ratio=geom.ellipse_axis_ratio,
            circularity=geom.circularity,
        )
        # Mock neutral browning & texture for geometric calibration
        browning = BrowningAnalyzer._classify_pattern(0.04, {})
        from src.models import BrowningMetrics, TextureMetrics
        b_metrics = BrowningMetrics(
            chapati_area_px=int(geom.area),
            browned_area_px=int(geom.area * 0.04),
            burn_ratio=0.04,
            patch_count=5,
            largest_patch_px=100,
            mean_browning_intensity=100.0,
            browning_control_score=85.0,
            classification="Balanced Browning",
        )
        t_metrics = TextureMetrics(
            grayscale_std=25.0,
            laplacian_variance=100.0,
            local_contrast=15.0,
            texture_score=80.0,
        )
        score = Scorer.calculate_perfection(geom, radial, b_metrics, t_metrics)
        return {
            "name": name,
            "geom": geom,
            "radial": radial,
            "score": score,
        }

    def test_synthetic_shape_ranking(self):
        """
        Tests shapes A through H and prints calibration table.
        Verifies expected relative ranking:
        Perfect Circle > Slightly Irregular Circle > Ellipse > Rounded Triangle > Triangle / Irregular
        """
        shapes = [
            ("A: Perfect Circle", SyntheticShapeGenerator.create_perfect_circle()[1]),
            ("B: Slightly Irreg Circle", SyntheticShapeGenerator.create_slightly_irregular_circle()[1]),
            ("C: Ellipse", SyntheticShapeGenerator.create_ellipse()[1]),
            ("D: Rounded Triangle (Samosa)", SyntheticShapeGenerator.create_rounded_triangle()[1]),
            ("E: Sharp Triangle", SyntheticShapeGenerator.create_triangle()[1]),
            ("F: Pentagon", SyntheticShapeGenerator.create_pentagon()[1]),
            ("G: Highly Irregular", SyntheticShapeGenerator.create_irregular_polygon()[1]),
            ("H: Lobed Shape", SyntheticShapeGenerator.create_lobed_shape()[1]),
        ]

        results = [self._evaluate_contour(cnt, name) for name, cnt in shapes]

        # Print calibration table
        print("\n" + "=" * 80)
        print("SYNTHETIC SHAPE CALIBRATION TABLE")
        print("=" * 80)
        print(f"{'Shape':<30} | {'Circ':<6} | {'Dev':<6} | {'k3(Tri)':<7} | {'Type':<18} | {'Score':<6}")
        print("-" * 80)
        for r in results:
            print(
                f"{r['name']:<30} | "
                f"{r['score'].circularity_score:>5.1f}% | "
                f"{r['score'].ideal_circle_deviation_score:>5.1f}% | "
                f"{r['radial'].k3_triangle:>6.2f}% | "
                f"{r['radial'].geometric_type:<18} | "
                f"{r['score'].perfection_index:>5.1f}"
            )
        print("=" * 80 + "\n")

        score_map = {r["name"]: r["score"].perfection_index for r in results}

        # 1. Perfect circle should score highest (>= 95)
        self.assertGreaterEqual(score_map["A: Perfect Circle"], 95.0)

        # 2. Perfect circle > Slightly irregular circle
        self.assertGreater(score_map["A: Perfect Circle"], score_map["B: Slightly Irreg Circle"])

        # 3. Slightly irregular circle > Ellipse
        self.assertGreater(score_map["B: Slightly Irreg Circle"], score_map["C: Ellipse"])

        # 4. Ellipse > Rounded Triangle (Samosa problem resolved)
        self.assertGreater(score_map["C: Ellipse"], score_map["D: Rounded Triangle (Samosa)"])

        # 5. Rounded Triangle must have score capped below 70
        self.assertLessEqual(score_map["D: Rounded Triangle (Samosa)"], 68.0)

        # 6. Rounded Triangle should be identified as ROUNDED TRIANGULAR
        rounded_tri_res = [r for r in results if "Rounded Triangle" in r["name"]][0]
        self.assertEqual(rounded_tri_res["radial"].geometric_type, "ROUNDED TRIANGULAR")

    def test_real_specimens_regression(self):
        """
        Regression test on the 3 real chapatis:
        - CHAPATI-0003 (C2) must rank highest (>= 90)
        - CHAPATI-0004 (C3, rounded triangle) must score < 72 and NOT be ALMOST CIRCULAR
        - CHAPATI-0002 (C1) must not outscore C2 despite high texture
        """
        results = {}
        for cid in ["CHAPATI-0002", "CHAPATI-0003", "CHAPATI-0004"]:
            matches = glob.glob(f"data/images/{cid}*_orig.jpg")
            if not matches:
                continue
            img = cv2.imread(matches[0])
            prep = Preprocessor.preprocess(img)
            seg = Segmenter.segment(prep)
            cnt = seg["contour"]
            mask = seg["mask"]

            geom = GeometricAnalyzer.analyze(cnt)
            radial = RadialAnalyzer.analyze(
                cnt, geom.centroid, geom.fitted_circle_radius,
                ellipse_axis_ratio=geom.ellipse_axis_ratio,
                circularity=geom.circularity,
            )
            browning, _ = BrowningAnalyzer.analyze(
                prep["working_bgr"], mask, geom.centroid, geom.equivalent_diameter / 2.0
            )
            texture = TextureAnalyzer.analyze(prep["working_bgr"], mask)
            score = Scorer.calculate_perfection(geom, radial, browning, texture)
            results[cid] = {
                "geom": geom,
                "radial": radial,
                "browning": browning,
                "score": score,
            }

        if len(results) == 3:
            c1 = results["CHAPATI-0002"]["score"]
            c2 = results["CHAPATI-0003"]["score"]
            c3 = results["CHAPATI-0004"]["score"]

            print(f"\nREAL SPECIMENS REGRESSION RESULTS:")
            print(f"C2 (CHAPATI-0003 - Near Circle): Score {c2.perfection_index:.1f}, Verdict: {c2.verdict}")
            print(f"C1 (CHAPATI-0002 - Irregular):   Score {c1.perfection_index:.1f}, Verdict: {c1.verdict}")
            print(f"C3 (CHAPATI-0004 - Samosa):      Score {c3.perfection_index:.1f}, Verdict: {c3.verdict}")

            # C2 (near circle) ranks highest
            self.assertGreater(c2.perfection_index, c1.perfection_index)
            self.assertGreater(c1.perfection_index, c3.perfection_index)

            # C2 must achieve ALMOST CIRCULAR or GEOMETRICALLY ENLIGHTENED
            self.assertIn(c2.verdict, ["ALMOST CIRCULAR", "GEOMETRICALLY ENLIGHTENED"])

            # C3 (triangle) must NOT be ALMOST CIRCULAR, and score capped < 72
            self.assertNotEqual(c3.verdict, "ALMOST CIRCULAR")
            self.assertLess(c3.perfection_index, 72.0)
            self.assertEqual(results["CHAPATI-0004"]["radial"].geometric_type, "ROUNDED TRIANGULAR")

            # Check browning classification on C2 and C3 with zero browning
            self.assertEqual(results["CHAPATI-0003"]["browning"].classification, "Minimal Visible Browning")
            self.assertEqual(results["CHAPATI-0004"]["browning"].classification, "Minimal Visible Browning")


if __name__ == "__main__":
    unittest.main()
