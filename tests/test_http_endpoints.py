"""
HTTP Integration tests for Flask API and PWA endpoints.
Verifies:
- GET / (serves HTML)
- GET /manifest.json (valid PWA manifest)
- GET /sw.js (Service Worker with Service-Worker-Allowed header)
- POST /api/analyze (valid image analysis, error handling)
- GET /api/history (returns list)
- GET /api/export/csv (returns CSV file)
- GET /api/export/json/<id> (returns structured JSON)
"""

import unittest
import json
import io
import cv2
from app import app
from tests.synthetic_shapes import SyntheticShapeGenerator


class TestHttpEndpoints(unittest.TestCase):

    def setUp(self):
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_pwa_shell_routes(self):
        # 1. HTML index
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"CHAPATI ANALYZER", res.data)

        # 2. Manifest
        res_m = self.client.get("/manifest.json")
        self.assertEqual(res_m.status_code, 200)
        manifest_json = json.loads(res_m.data.decode("utf-8"))
        self.assertEqual(manifest_json["short_name"], "ChapatiLab")
        self.assertEqual(manifest_json["display"], "standalone")

        # 3. Service worker
        res_sw = self.client.get("/sw.js")
        self.assertEqual(res_sw.status_code, 200)
        self.assertEqual(res_sw.headers.get("Service-Worker-Allowed"), "/")

    def test_api_analyze_and_exports(self):
        # Generate synthetic chapati image
        img_bgr = SyntheticShapeGenerator.create_synthetic_chapati_image(radius=150, img_size=400, with_browning=True)
        _, encoded = cv2.imencode(".jpg", img_bgr)
        image_bytes = encoded.tobytes()

        # POST multipart/form-data
        data = {
            "image": (io.BytesIO(image_bytes), "specimen.jpg")
        }
        res = self.client.post("/api/analyze", data=data, content_type="multipart/form-data")
        self.assertEqual(res.status_code, 200)

        res_json = json.loads(res.data.decode("utf-8"))
        self.assertTrue(res_json["success"])
        specimen_id = res_json["data"]["id"]
        self.assertTrue(specimen_id.startswith("CHAPATI-"))

        # Test history endpoint
        res_hist = self.client.get("/api/history")
        self.assertEqual(res_hist.status_code, 200)
        hist_json = json.loads(res_hist.data.decode("utf-8"))
        self.assertTrue(hist_json["success"])
        self.assertGreaterEqual(hist_json["count"], 1)

        # Test CSV export
        res_csv = self.client.get("/api/export/csv")
        self.assertEqual(res_csv.status_code, 200)
        self.assertIn(b"CHAPATI-", res_csv.data)

        # Test JSON export
        res_export = self.client.get(f"/api/export/json/{specimen_id}")
        self.assertEqual(res_export.status_code, 200)
        export_json = json.loads(res_export.data.decode("utf-8"))
        self.assertEqual(export_json["id"], specimen_id)
        self.assertIn("geometry", export_json)
        self.assertIn("score", export_json)

    def test_empty_payload_error_handling(self):
        res = self.client.post("/api/analyze", data={})
        self.assertEqual(res.status_code, 400)
        err = json.loads(res.data.decode("utf-8"))
        self.assertFalse(err["success"])
        self.assertEqual(err["code"], "EMPTY_PAYLOAD")


if __name__ == "__main__":
    unittest.main()
