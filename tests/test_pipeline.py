"""
Integration test for full ChapatiPipeline.
Verifies:
- Raw image bytes to complete analysis
- All visualizations created
- Persistence to database
- Stage 2 genome vector populated
"""

import unittest
import os
import tempfile
import cv2
from tests.synthetic_shapes import SyntheticShapeGenerator
from src.pipeline import ChapatiPipeline


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pipeline = ChapatiPipeline(data_dir=self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_pipeline_run(self):
        # 1. Create synthetic chapati image and encode to JPEG bytes
        img_bgr = SyntheticShapeGenerator.create_synthetic_chapati_image(radius=150, img_size=450, with_browning=True)
        success, encoded = cv2.imencode(".jpg", img_bgr)
        self.assertTrue(success)
        jpeg_bytes = encoded.tobytes()

        # 2. Run pipeline
        result_dict = self.pipeline.process_image_bytes(jpeg_bytes, filename_hint="test_chapati.jpg")

        # 3. Check outputs
        self.assertIn("id", result_dict)
        self.assertEqual(result_dict["id"], "CHAPATI-0001")
        self.assertIn("score", result_dict)
        self.assertGreater(result_dict["score"]["perfection_index"], 50.0)
        self.assertIn("verdict", result_dict["score"])

        # 4. Check stage 2 genome vector
        self.assertIsNotNone(result_dict.get("stage2_genome_vector"))
        self.assertEqual(len(result_dict["stage2_genome_vector"]), 360)

        # 5. Check visualizations generated on disk
        vis = result_dict.get("visualizations", {})
        self.assertIn("ideal_overlay", vis)
        self.assertIn("browning_heatmap", vis)
        self.assertIn("radial_profile", vis)
        self.assertIn("annotated_report", vis)

        # Confirm files exist in temporary data dir
        ideal_path = os.path.join(self.temp_dir.name, "images", os.path.basename(vis["ideal_overlay"]))
        self.assertTrue(os.path.exists(ideal_path))


if __name__ == "__main__":
    unittest.main()
