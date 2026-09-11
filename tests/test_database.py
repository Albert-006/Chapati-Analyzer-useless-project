"""
Unit tests for SQLite database operations, ID generation, and exports.
"""

import unittest
import os
import tempfile
from src.database import Database
from src.models import (
    ChapatiAnalysisResult,
    GeometricMetrics,
    RadialMetrics,
    BrowningMetrics,
    TextureMetrics,
    PerfectionScore,
)


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_chapati.db")
        self.db = Database(self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _sample_result(self, analysis_id="CHAPATI-0001"):
        geom = GeometricMetrics(
            area=50000.0, perimeter=800.0, circularity=0.91,
            circularity_raw=0.91, circularity_light=0.91, circularity_moderate=0.91,
            bounding_box=(50, 50, 250, 250), width=250.0, height=250.0,
            aspect_ratio=1.0, equivalent_diameter=252.0, solidity=0.99,
            ellipse_major_axis=252.0, ellipse_minor_axis=250.0,
            ellipse_axis_ratio=1.008, ellipse_eccentricity=0.12,
            ellipse_roundness_score=98.0, centroid=(175.0, 175.0),
            bbox_center=(175.0, 175.0), fitted_circle_center=(175.0, 175.0),
            fitted_circle_radius=126.0, center_deviation_px=0.0,
            center_deviation_normalized=0.0, center_accuracy_score=98.0
        )
        radial = RadialMetrics(
            angles_deg=list(range(360)), radii=[126.0]*360, mean_radius=126.0,
            min_radius=126.0, max_radius=126.0, std_radius=1.5,
            coefficient_of_variation=0.012, range_over_mean=0.02,
            median_absolute_deviation=1.0, max_radial_deviation_normalized=0.015,
            radius_stability_score=92.0, symmetry_diff_180=0.02, symmetry_score=94.0,
            boundary_roughness=0.005, boundary_smoothness_score=93.0,
            ideal_circle_mean_error=0.01, ideal_circle_rms_error=0.015,
            ideal_circle_max_error=0.02, ideal_circle_p95_error=0.018,
            ideal_circle_deviation_score=92.0,
            k2_ellipse=0.8, k3_triangle=0.5, k4_lobed=0.4,
            harmonic_irregularity=1.5, harmonic_score=95.0,
            corner_count=0, corner_strength="LOW", max_turning_angle=45.0,
            corner_quality_score=95.0, geometric_type="NEAR CIRCLE"
        )
        browning = BrowningMetrics(
            chapati_area_px=50000, browned_area_px=2500, burn_ratio=0.05,
            patch_count=8, largest_patch_px=200, mean_browning_intensity=120.0,
            browning_control_score=90.0, classification="Balanced Browning"
        )
        texture = TextureMetrics(
            grayscale_std=25.0, laplacian_variance=120.0, local_contrast=15.0,
            texture_score=85.0
        )
        score = PerfectionScore(
            circularity_score=91.0, ideal_circle_deviation_score=92.0,
            radius_stability_score=92.0, ellipse_roundness_score=98.0,
            corner_quality_score=95.0, symmetry_score=94.0,
            boundary_smoothness_score=93.0, center_accuracy_score=98.0,
            browning_control_score=90.0, texture_score=85.0,
            raw_weighted_score=92.4, perfection_index=92.4,
            geometric_type="NEAR CIRCLE",
            verdict="ALMOST CIRCULAR", verdict_quote="Grandma would probably approve."
        )
        return ChapatiAnalysisResult(
            id=analysis_id,
            timestamp="2026-09-11 23:00:00",
            image_filename="sample.jpg",
            geometry=geom,
            radial=radial,
            browning=browning,
            texture=texture,
            score=score,
            stage2_genome_vector=[1.0]*360,
        )

    def test_id_generation(self):
        id1 = self.db.generate_next_id()
        self.assertEqual(id1, "CHAPATI-0001")

        result1 = self._sample_result(id1)
        self.db.save_analysis(result1)

        id2 = self.db.generate_next_id()
        self.assertEqual(id2, "CHAPATI-0002")

    def test_save_and_retrieve(self):
        res = self._sample_result("CHAPATI-0001")
        self.db.save_analysis(res)

        loaded = self.db.get_analysis_by_id("CHAPATI-0001")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["id"], "CHAPATI-0001")
        self.assertEqual(loaded["score"]["perfection_index"], 92.4)
        self.assertEqual(loaded["score"]["verdict"], "ALMOST CIRCULAR")

    def test_history_summaries(self):
        summaries = self.db.get_all_summaries()
        self.assertEqual(len(summaries), 0)

        res = self._sample_result("CHAPATI-0001")
        self.db.save_analysis(res)
        summaries = self.db.get_all_summaries()
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["id"], "CHAPATI-0001")

    def test_csv_export(self):
        res = self._sample_result("CHAPATI-0001")
        self.db.save_analysis(res)

        csv_text = self.db.export_csv()
        self.assertIn("CHAPATI-0001", csv_text)
        self.assertIn("ALMOST CIRCULAR", csv_text)
        self.assertIn("Circularity", csv_text)


if __name__ == "__main__":
    unittest.main()
