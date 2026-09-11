"""
Unit tests for browning and texture analysis.
Verifies PRD Section 18, 19, 20:
- Browning mask, burn ratio, patch count, largest patch
- 5 concentric radial zones
- Browning pattern classification
- Visual texture analysis
"""

import unittest
import numpy as np
import cv2
from tests.synthetic_shapes import SyntheticShapeGenerator
from src.browning_analysis import BrowningAnalyzer
from src.texture_analysis import TextureAnalyzer


class TestBrowningAndTexture(unittest.TestCase):

    def test_browning_zones_and_burn_ratio(self):
        # Create a synthetic image with browning spots
        img_bgr = SyntheticShapeGenerator.create_synthetic_chapati_image(radius=150, img_size=400, with_browning=True)

        # Generate chapati mask
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        centroid = (200.0, 200.0)
        metrics, browning_mask = BrowningAnalyzer.analyze(img_bgr, mask, centroid, equivalent_radius=150.0)

        self.assertGreater(metrics.chapati_area_px, 10000)
        self.assertGreater(metrics.browned_area_px, 0)
        self.assertGreater(metrics.burn_ratio, 0.0)
        self.assertLess(metrics.burn_ratio, 0.40)
        self.assertGreater(metrics.patch_count, 0)
        self.assertGreater(metrics.largest_patch_px, 0)

        # 5 concentric zones verified
        self.assertEqual(len(metrics.zones), 5)
        zone_names = [z.name for z in metrics.zones]
        self.assertEqual(zone_names, ["Center", "Inner", "Middle", "Outer", "Edge"])

        # Every zone must have non-negative area
        for z in metrics.zones:
            self.assertGreater(z.area_px, 0)
            self.assertGreaterEqual(z.browning_density, 0.0)

    def test_texture_analysis(self):
        img_bgr = SyntheticShapeGenerator.create_synthetic_chapati_image(radius=150, img_size=400, with_browning=True)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

        texture = TextureAnalyzer.analyze(img_bgr, mask)

        self.assertGreater(texture.grayscale_std, 0.0)
        self.assertGreater(texture.laplacian_variance, 0.0)
        self.assertGreater(texture.local_contrast, 0.0)
        self.assertGreaterEqual(texture.texture_score, 0.0)
        self.assertLessEqual(texture.texture_score, 100.0)
        self.assertIn("Measures visible image surface texture", texture.disclaimer)


if __name__ == "__main__":
    unittest.main()
