"""
Visualization Module for Chapati Analyzer™.
Generates technical laboratory visualizations:
- Contour vs. Ideal Circle overlay ("The circle your chapati was trying to be")
- Radial profile plot r(theta) with ideal circle residual shading
- Browning heatmap with concentric zones
- Complete annotated laboratory report card with Shape Diagnostics
"""

from typing import Tuple, Dict, Any
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from src.models import ChapatiAnalysisResult


class Visualizer:
    """Generates visual analysis artifacts and technical diagrams."""

    COLOR_CONTOUR = (46, 204, 113)       # Emerald green (BGR)
    COLOR_IDEAL_CIRCLE = (0, 165, 255)   # Amber / Gold (BGR)
    COLOR_CENTROID = (255, 105, 180)     # Neon Pink (BGR)
    COLOR_CIRCLE_CENTER = (255, 215, 0)  # Cyan/Gold (BGR)

    @classmethod
    def create_ideal_circle_overlay(
        cls,
        image_bgr: np.ndarray,
        contour: np.ndarray,
        centroid: Tuple[float, float],
        fitted_circle: Tuple[float, float, float],
    ) -> np.ndarray:
        """
        Draws the detected boundary in emerald green and the ideal fitted circle
        in glowing amber with centroid and offset indicator.
        """
        overlay = image_bgr.copy()
        fx, fy, fr = fitted_circle
        cx, cy = centroid

        # 1. Draw ideal circle
        center_int = (int(round(fx)), int(round(fy)))
        radius_int = int(round(fr))
        cv2.circle(overlay, center_int, radius_int, cls.COLOR_IDEAL_CIRCLE, thickness=2, lineType=cv2.LINE_AA)

        # 2. Draw actual contour
        cv2.drawContours(overlay, [contour], -1, cls.COLOR_CONTOUR, thickness=3, lineType=cv2.LINE_AA)

        # 3. Draw centroid and fitted circle center
        cv2.drawMarker(
            overlay, (int(round(cx)), int(round(cy))), cls.COLOR_CENTROID,
            markerType=cv2.MARKER_TILTED_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA
        )
        cv2.circle(overlay, center_int, 4, cls.COLOR_CIRCLE_CENTER, thickness=-1, lineType=cv2.LINE_AA)

        # 4. Draw line between centers showing displacement
        cv2.line(
            overlay, (int(round(cx)), int(round(cy))), center_int,
            (0, 0, 255), thickness=2, lineType=cv2.LINE_AA
        )

        # 5. Legend HUD in corner
        hud_bg = overlay.copy()
        cv2.rectangle(hud_bg, (10, 10), (320, 95), (15, 23, 42), -1)
        overlay = cv2.addWeighted(hud_bg, 0.75, overlay, 0.25, 0)

        cv2.putText(overlay, "CHAPATI BOUNDARY", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (46, 204, 113), 1, cv2.LINE_AA)
        cv2.putText(overlay, "IDEAL FITTED CIRCLE", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 1, cv2.LINE_AA)
        cv2.putText(overlay, "CENTER DEVIATION VECTOR", (25, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1, cv2.LINE_AA)

        return overlay

    @classmethod
    def create_radial_profile_plot(cls, result: ChapatiAnalysisResult, output_path: str) -> str:
        """
        Plots the radial function r(theta) vs theta (0 to 359 degrees) against
        the ideal circle radius, highlighting residual deviations.
        """
        angles = np.array(result.radial.angles_deg)
        radii = np.array(result.radial.radii)
        r_ideal = result.geometry.fitted_circle_radius
        mean_r = result.radial.mean_radius
        rms_err_pct = result.radial.ideal_circle_rms_error * 100.0

        plt.figure(figsize=(7.5, 4.0), dpi=150, facecolor="#0F172A")
        ax = plt.subplot(111, facecolor="#1E293B")

        # Deviation shading between actual and ideal
        ax.fill_between(
            angles, radii, r_ideal,
            where=(radii >= r_ideal), color="#38BDF8", alpha=0.25, label="Positive Radial Deviation"
        )
        ax.fill_between(
            angles, radii, r_ideal,
            where=(radii < r_ideal), color="#F43F5E", alpha=0.25, label="Negative Radial Deviation"
        )

        # Plot curves
        ax.plot(angles, radii, color="#38BDF8", linewidth=2.0, label="Actual Contour r(θ)")
        ax.axhline(r_ideal, color="#F59E0B", linestyle="-", linewidth=2.0, label=f"Ideal Circle ({r_ideal:.1f}px)")
        ax.axhline(mean_r, color="#94A3B8", linestyle="--", linewidth=1.2, label=f"Mean Radius ({mean_r:.1f}px)")

        title_text = f"Radial Signal & Ideal Circle Residual | RMS Error: {rms_err_pct:.1f}%"
        ax.set_title(title_text, color="#F8FAFC", fontsize=11, fontweight="bold", pad=12)
        ax.set_xlabel("Angle θ (degrees)", color="#94A3B8", fontsize=9)
        ax.set_ylabel("Radius (px)", color="#94A3B8", fontsize=9)

        ax.set_xlim(0, 359)
        ax.set_xticks(np.arange(0, 361, 45))
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748B")

        ax.tick_params(colors="#94A3B8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")

        leg = ax.legend(loc="upper right", facecolor="#0F172A", edgecolor="#334155", fontsize=7.5)
        for text in leg.get_texts():
            text.set_color("#E2E8F0")

        plt.tight_layout()
        plt.savefig(output_path, facecolor="#0F172A", edgecolor="none")
        plt.close()
        return output_path

    @classmethod
    def create_browning_heatmap_overlay(
        cls,
        image_bgr: np.ndarray,
        browning_mask: np.ndarray,
        chapati_mask: np.ndarray,
        centroid: Tuple[float, float],
        equivalent_radius: float,
    ) -> np.ndarray:
        """
        Creates a thermal heatmap overlay representing browning density and concentric zones.
        """
        overlay = image_bgr.copy()
        cx, cy = centroid

        density = cv2.GaussianBlur(browning_mask.astype(np.float32), (35, 35), 0)
        max_val = np.max(density)
        norm_density = np.uint8((density / max_val) * 255) if max_val > 0 else np.zeros_like(browning_mask)

        heatmap_color = cv2.applyColorMap(norm_density, cv2.COLORMAP_HOT)

        mask_3ch = cv2.merge([chapati_mask, chapati_mask, chapati_mask]) > 0
        blended = cv2.addWeighted(overlay, 0.65, heatmap_color, 0.35, 0)
        overlay[mask_3ch] = blended[mask_3ch]

        zone_radii = [0.20, 0.40, 0.60, 0.80, 1.00]
        for frac in zone_radii:
            r_px = int(round(frac * equivalent_radius))
            cv2.circle(
                overlay, (int(round(cx)), int(round(cy))), r_px,
                (200, 200, 255), thickness=1, lineType=cv2.LINE_AA
            )

        return overlay

    @classmethod
    def create_comprehensive_annotated_card(
        cls,
        original_bgr: np.ndarray,
        ideal_overlay: np.ndarray,
        heatmap_overlay: np.ndarray,
        result: ChapatiAnalysisResult,
        output_path: str,
    ) -> str:
        """
        Assembles a certified laboratory analysis summary card with metrics and visual overlays.
        """
        h_img, w_img = ideal_overlay.shape[:2]
        target_w = 420
        scale = target_w / float(w_img)
        target_h = int(round(h_img * scale))

        thumb1 = cv2.resize(original_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA)
        thumb2 = cv2.resize(ideal_overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)
        thumb3 = cv2.resize(heatmap_overlay, (target_w, target_h), interpolation=cv2.INTER_AREA)

        card_w = target_w * 3 + 40
        card_h = target_h + 290
        canvas = np.full((card_h, card_w, 3), (15, 23, 42), dtype=np.uint8)  # Slate dark #0F172A

        # Header
        cv2.putText(canvas, "CHAPATI ANALYZER™ LABORATORY REPORT", (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.85, (245, 158, 11), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"SPECIMEN: {result.id}  |  TYPE: {result.radial.geometric_type}  |  {result.timestamp}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (148, 163, 184), 1, cv2.LINE_AA)

        # Paste 3 thumbnails
        y_top = 95
        canvas[y_top:y_top + target_h, 15:15 + target_w] = thumb1
        canvas[y_top:y_top + target_h, 25 + target_w:25 + target_w * 2] = thumb2
        canvas[y_top:y_top + target_h, 35 + target_w * 2:35 + target_w * 3] = thumb3

        # Subtitles under images
        y_sub = y_top + target_h + 20
        cv2.putText(canvas, "ORIGINAL SPECIMEN", (15, y_sub), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(canvas, "IDEAL CIRCLE vs ACTUAL", (25 + target_w, y_sub), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
        cv2.putText(canvas, "BROWNING CONCENTRIC ZONES", (35 + target_w * 2, y_sub), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

        # Score & Diagnostics Banner
        y_score = y_sub + 35
        cv2.rectangle(canvas, (15, y_score), (card_w - 15, y_score + 105), (30, 41, 59), -1)

        cv2.putText(
            canvas, f"PERFECTION INDEX: {result.score.perfection_index:.1f} / 100",
            (30, y_score + 32), cv2.FONT_HERSHEY_DUPLEX, 0.75, (245, 158, 11), 2, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"GEOMETRIC TYPE: {result.radial.geometric_type}  |  CORNER STRENGTH: {result.radial.corner_strength}",
            (30, y_score + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA
        )
        cv2.putText(
            canvas, f"VERDICT: {result.score.verdict} — \"{result.score.verdict_quote}\"",
            (30, y_score + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA
        )

        cv2.imwrite(output_path, canvas)
        return output_path
