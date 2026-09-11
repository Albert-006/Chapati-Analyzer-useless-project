"""
End-to-End Pipeline Orchestrator for Chapati Analyzer™.
Integrates preprocessing, segmentation, geometric analysis, radial sampling,
browning quantification, visual texture, scoring, visualization, and persistence.
"""

from typing import Dict, Any, Tuple
import os
import datetime
import cv2
import numpy as np

from src.models import ChapatiAnalysisResult
from src.preprocessing import Preprocessor, ImageValidationError
from src.segmentation import Segmenter, ChapatiDetectionError
from src.geometric_analysis import GeometricAnalyzer
from src.radial_analysis import RadialAnalyzer
from src.browning_analysis import BrowningAnalyzer
from src.texture_analysis import TextureAnalyzer
from src.scoring import Scorer
from src.visualization import Visualizer
from src.database import Database


class ChapatiPipeline:
    """Orchestrates the complete computer-vision analysis pipeline for a chapati image."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        self.db = Database(os.path.join(data_dir, "chapati.db"))

    def process_image_bytes(self, image_bytes: bytes, filename_hint: str = "capture.jpg") -> Dict[str, Any]:
        """
        Processes raw bytes from file upload or camera capture.
        Returns a rich dictionary containing the analysis result and generated visualization paths.
        """
        # 1. Load and validate
        raw_bgr = Preprocessor.load_and_validate(image_bytes)

        # 2. Preprocess
        prep = Preprocessor.preprocess(raw_bgr)
        working_bgr = prep["working_bgr"]

        # 3. Detect & Segment
        seg = Segmenter.segment(prep)
        contour = seg["contour"]
        chapati_mask = seg["mask"]

        # 4. Geometric analysis
        geom = GeometricAnalyzer.analyze(contour)

        # 5. Radial, Symmetry, Ideal Circle Residual & Fourier analysis
        radial = RadialAnalyzer.analyze(
            contour,
            geom.centroid,
            geom.fitted_circle_radius,
            ellipse_axis_ratio=geom.ellipse_axis_ratio,
            circularity=geom.circularity,
        )

        # 6. Browning analysis
        eq_radius = geom.equivalent_diameter / 2.0
        browning, browning_mask = BrowningAnalyzer.analyze(
            working_bgr, chapati_mask, geom.centroid, eq_radius
        )

        # 7. Visual texture analysis
        texture = TextureAnalyzer.analyze(working_bgr, chapati_mask)

        # 8. Perfection Score & Verdict
        score = Scorer.calculate_perfection(geom, radial, browning, texture)

        # 9. Next specimen ID & timestamp
        analysis_id = self.db.generate_next_id()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 10. Future Stage 2: Normalized Shape Genome representation
        # 360-bin radius profile normalized by mean radius (scale-invariant fingerprint)
        stage2_genome_vector = [
            round(float(r / geom.fitted_circle_radius), 4) for r in radial.radii
        ]

        # 11. Construct Result Object
        base_name = f"{analysis_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        original_filename = f"{base_name}_orig.jpg"
        ideal_filename = f"{base_name}_ideal.jpg"
        heatmap_filename = f"{base_name}_browning.jpg"
        profile_filename = f"{base_name}_profile.png"
        annotated_card_filename = f"{base_name}_report.jpg"

        result = ChapatiAnalysisResult(
            id=analysis_id,
            timestamp=timestamp,
            image_filename=original_filename,
            geometry=geom,
            radial=radial,
            browning=browning,
            texture=texture,
            score=score,
            stage2_genome_vector=stage2_genome_vector,
        )

        # 12. Generate and save visualizations
        orig_save_path = os.path.join(self.images_dir, original_filename)
        cv2.imwrite(orig_save_path, prep["original_bgr"])

        # Ideal circle overlay
        ideal_overlay = Visualizer.create_ideal_circle_overlay(
            working_bgr, contour, geom.centroid,
            (geom.fitted_circle_center[0], geom.fitted_circle_center[1], geom.fitted_circle_radius)
        )
        ideal_save_path = os.path.join(self.images_dir, ideal_filename)
        cv2.imwrite(ideal_save_path, ideal_overlay)

        # Browning heatmap overlay
        heatmap_overlay = Visualizer.create_browning_heatmap_overlay(
            working_bgr, browning_mask, chapati_mask, geom.centroid, eq_radius
        )
        heatmap_save_path = os.path.join(self.images_dir, heatmap_filename)
        cv2.imwrite(heatmap_save_path, heatmap_overlay)

        # Radial profile graph
        profile_save_path = os.path.join(self.images_dir, profile_filename)
        Visualizer.create_radial_profile_plot(result, profile_save_path)

        # Full laboratory report card
        report_save_path = os.path.join(self.images_dir, annotated_card_filename)
        Visualizer.create_comprehensive_annotated_card(
            working_bgr, ideal_overlay, heatmap_overlay, result, report_save_path
        )

        # 13. Persist into SQLite
        self.db.save_analysis(result)

        # 14. Return structured response
        response_data = result.to_dict()
        response_data["visualizations"] = {
            "original": f"/data/images/{original_filename}",
            "ideal_overlay": f"/data/images/{ideal_filename}",
            "browning_heatmap": f"/data/images/{heatmap_filename}",
            "radial_profile": f"/data/images/{profile_filename}",
            "annotated_report": f"/data/images/{annotated_card_filename}",
        }

        return response_data
