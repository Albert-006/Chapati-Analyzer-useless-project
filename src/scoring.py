"""
Scoring and Verdict Module for Chapati Analyzer™.
Combines geometric, radial residual, Fourier harmonic, and texture metrics
into the Chapati Perfection Index™ using a geometry-first weighting system
and hard geometric sanity gates.
"""

from typing import Tuple, Optional
from src.models import (
    GeometricMetrics,
    RadialMetrics,
    BrowningMetrics,
    TextureMetrics,
    PerfectionScore,
)


class Scorer:
    """Computes the Chapati Perfection Index™ and assigns verdicts with geometric sanity gates."""

    # Geometry-first weights (Geometry accounts for 92% of score)
    WEIGHT_CIRCULARITY = 0.20
    WEIGHT_IDEAL_CIRCLE_DEVIATION = 0.20
    WEIGHT_RADIUS_STABILITY = 0.15
    WEIGHT_ELLIPSE_ROUNDNESS = 0.10
    WEIGHT_CORNER_QUALITY = 0.10
    WEIGHT_SYMMETRY = 0.10
    WEIGHT_BOUNDARY_SMOOTHNESS = 0.07
    WEIGHT_CENTER_ACCURACY = 0.03
    WEIGHT_BROWNING_CONTROL = 0.03
    WEIGHT_TEXTURE = 0.02

    @classmethod
    def calculate_perfection(
        cls,
        geometry: GeometricMetrics,
        radial: RadialMetrics,
        browning: BrowningMetrics,
        texture: TextureMetrics,
    ) -> PerfectionScore:
        """
        Calculates the weighted perfection index and applies geometric sanity gates.
        Guaranteed 100% deterministic with zero randomness.
        """
        # Sub-scores scaled to 0-100
        circ_score = float(geometry.circularity * 100.0)
        dev_score = float(radial.ideal_circle_deviation_score)
        stab_score = float(radial.radius_stability_score)
        ellipse_score = float(geometry.ellipse_roundness_score)
        corner_score = float(radial.corner_quality_score)
        symm_score = float(radial.symmetry_score)
        smooth_score = float(radial.boundary_smoothness_score)
        center_score = float(geometry.center_accuracy_score)
        browning_score = float(browning.browning_control_score)
        text_score = float(texture.texture_score)

        # Weighted calculation
        raw_weighted_score = (
            cls.WEIGHT_CIRCULARITY * circ_score +
            cls.WEIGHT_IDEAL_CIRCLE_DEVIATION * dev_score +
            cls.WEIGHT_RADIUS_STABILITY * stab_score +
            cls.WEIGHT_ELLIPSE_ROUNDNESS * ellipse_score +
            cls.WEIGHT_CORNER_QUALITY * corner_score +
            cls.WEIGHT_SYMMETRY * symm_score +
            cls.WEIGHT_BOUNDARY_SMOOTHNESS * smooth_score +
            cls.WEIGHT_CENTER_ACCURACY * center_score +
            cls.WEIGHT_BROWNING_CONTROL * browning_score +
            cls.WEIGHT_TEXTURE * text_score
        )

        raw_score = float(min(100.0, max(0.0, raw_weighted_score)))

        # Geometric Quality Gate: Prevent secondary metrics or high 4piA/P^2 from masking non-circular shapes
        capped_score, cap_applied, cap_reason = cls._apply_geometric_gate(
            raw_score=raw_score,
            radial=radial,
            geometry=geometry,
        )

        final_index = round(capped_score, 1)

        verdict, quote = cls.get_verdict(final_index, radial.geometric_type)

        return PerfectionScore(
            circularity_score=round(circ_score, 1),
            ideal_circle_deviation_score=round(dev_score, 1),
            radius_stability_score=round(stab_score, 1),
            ellipse_roundness_score=round(ellipse_score, 1),
            corner_quality_score=round(corner_score, 1),
            symmetry_score=round(symm_score, 1),
            boundary_smoothness_score=round(smooth_score, 1),
            center_accuracy_score=round(center_score, 1),
            browning_control_score=round(browning_score, 1),
            texture_score=round(text_score, 1),
            raw_weighted_score=round(raw_score, 1),
            perfection_index=final_index,
            geometric_type=radial.geometric_type,
            verdict=verdict,
            verdict_quote=quote,
            score_cap_applied=cap_applied,
            score_cap_reason=cap_reason,
        )

    @classmethod
    def _apply_geometric_gate(
        cls, raw_score: float, radial: RadialMetrics, geometry: GeometricMetrics
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Hard geometric quality gate that caps the maximum allowable Perfection Index
        if prominent non-circular defects are detected.
        """
        score = raw_score
        applied = False
        reason = None

        # Level 1: Severe non-circular defects (Triangle / Samosa / Extreme Ellipse / High Harmonic Irregularity)
        if radial.geometric_type == "ROUNDED TRIANGULAR" or radial.corner_strength == "HIGH":
            max_allowed = 68.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = "Severe non-circular structure detected (distinct triangular/corner geometry)."

        elif radial.geometric_type == "HIGHLY IRREGULAR":
            max_allowed = 65.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = "Severe geometric distortion: boundary deviates substantially from circular contour."

        elif geometry.ellipse_axis_ratio >= 1.35:
            max_allowed = 68.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = f"Severe elongation defect (axis ratio {geometry.ellipse_axis_ratio:.2f})."

        # Level 2: Moderate non-circular defects (Oval / Multi-lobed / High RMS error)
        elif radial.geometric_type == "OVAL" or geometry.ellipse_axis_ratio >= 1.12:
            max_allowed = 76.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = f"Moderate oval eccentricity (axis ratio {geometry.ellipse_axis_ratio:.2f})."

        elif radial.ideal_circle_rms_error >= 0.065:
            max_allowed = 74.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = f"High radial residual from ideal circle (RMS error {radial.ideal_circle_rms_error * 100:.1f}%)."

        # Level 3: Minor geometric anomalies
        elif radial.ideal_circle_rms_error >= 0.042 and radial.geometric_type != "NEAR CIRCLE":
            max_allowed = 86.0
            if score > max_allowed:
                score = max_allowed
                applied = True
                reason = "Noticeable boundary deviation from ideal circle."

        return score, applied, reason

    @staticmethod
    def get_verdict(score: float, geometric_type: str = "ROUND / IRREGULAR") -> Tuple[str, str]:
        """
        Maps numerical score and geometric classification to certified verdicts.
        Prevents non-circular shapes from receiving circularity approval verdicts.
        """
        # Samosa / Triangle guard: Never award Grandma approval to a triangle
        if geometric_type == "ROUNDED TRIANGULAR":
            if score < 65.0:
                return (
                    "SHAPE INCIDENT",
                    "Three distinct vertices detected. Samosa tendencies are mathematically undeniable.",
                )
            return (
                "QUESTIONABLE GEOMETRY",
                "The circle was merely a suggestion. A triangular geometry has emerged.",
            )

        if score >= 95.0:
            return (
                "GEOMETRICALLY ENLIGHTENED",
                "This chapati has achieved a level of circularity that may concern mathematicians.",
            )
        elif score >= 90.0:
            return (
                "ALMOST CIRCULAR",
                "Grandma would probably approve.",
            )
        elif score >= 80.0:
            return (
                "ACCEPTABLE CHAPATI",
                "Slightly geographically confused, but operational.",
            )
        elif score >= 70.0:
            return (
                "QUESTIONABLE GEOMETRY",
                "The circle was merely a suggestion.",
            )
        elif score >= 60.0:
            return (
                "SHAPE INCIDENT",
                "Something happened here.",
            )
        else:
            return (
                "PLEASE CONSULT A CHAPATI ENGINEER",
                "Immediate geometric intervention recommended.",
            )
