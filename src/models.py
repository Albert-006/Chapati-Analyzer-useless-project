"""
Data structures and dataclasses for Chapati Analyzer™.
Designed to be modular, typed, and forward-compatible with
Future Stage 2: Chapati Shape Genome™.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class GeometricMetrics:
    area: float  # Contour area in px^2
    perimeter: float  # Perimeter in px
    circularity: float  # 4 * pi * area / perimeter^2 (0.0 to 1.0)
    circularity_raw: float  # Circularity without smoothing
    circularity_light: float  # Circularity with light approximation
    circularity_moderate: float  # Circularity with moderate approximation
    bounding_box: Tuple[int, int, int, int]  # (x, y, w, h)
    width: float
    height: float
    aspect_ratio: float  # width / height
    equivalent_diameter: float  # 2 * sqrt(area / pi)
    solidity: float  # contour_area / convex_hull_area
    ellipse_major_axis: float
    ellipse_minor_axis: float
    ellipse_axis_ratio: float  # major / minor
    ellipse_eccentricity: float  # sqrt(1 - (minor/major)^2)
    ellipse_roundness_score: float  # 0 to 100
    centroid: Tuple[float, float]  # (cx, cy)
    bbox_center: Tuple[float, float]  # (x + w/2, y + h/2)
    fitted_circle_center: Tuple[float, float]  # (fx, fy)
    fitted_circle_radius: float
    center_deviation_px: float  # Euclidean distance between centroid and fitted circle center
    center_deviation_normalized: float  # deviation / fitted_radius
    center_accuracy_score: float  # 0 to 100


@dataclass
class RadialMetrics:
    angles_deg: List[float]  # 0 to 359
    radii: List[float]  # r(theta) for each angle
    mean_radius: float
    min_radius: float
    max_radius: float
    std_radius: float
    coefficient_of_variation: float  # std / mean
    range_over_mean: float  # (max - min) / mean
    median_absolute_deviation: float  # MAD of radius
    max_radial_deviation_normalized: float  # max(|r - mean|) / mean
    radius_stability_score: float  # 0 to 100
    symmetry_diff_180: float  # mean absolute difference between r(theta) and r(theta + 180)
    symmetry_score: float  # 0 to 100
    boundary_roughness: float  # residual standard deviation against smoothed signal
    boundary_smoothness_score: float  # 0 to 100 (renamed from edge smoothness)
    # Ideal Circle Residual Analysis
    ideal_circle_mean_error: float
    ideal_circle_rms_error: float
    ideal_circle_max_error: float
    ideal_circle_p95_error: float
    ideal_circle_deviation_score: float  # 0 to 100
    # Fourier Harmonic Analysis
    k2_ellipse: float
    k3_triangle: float
    k4_lobed: float
    harmonic_irregularity: float
    harmonic_score: float  # 0 to 100
    # Corner / Vertex Analysis
    corner_count: int
    corner_strength: str  # 'LOW', 'MODERATE', 'HIGH'
    max_turning_angle: float
    corner_quality_score: float  # 0 to 100
    # Overall Geometric Classification
    geometric_type: str  # 'PERFECT CIRCLE', 'NEAR CIRCLE', 'ROUNDED TRIANGULAR', 'OVAL', etc.


@dataclass
class BrowningZone:
    name: str  # 'Center', 'Inner', 'Middle', 'Outer', 'Edge'
    radius_fraction_min: float
    radius_fraction_max: float
    area_px: int
    browned_pixels: int
    browning_density: float  # browned / area (0.0 to 1.0)
    mean_intensity: float  # 0.0 to 255.0


@dataclass
class BrowningMetrics:
    chapati_area_px: int
    browned_area_px: int
    burn_ratio: float  # browned_area / chapati_area
    patch_count: int
    largest_patch_px: int
    mean_browning_intensity: float
    browning_control_score: float  # 0 to 100
    classification: str  # e.g., 'Minimal Visible Browning', 'Balanced Browning'
    zones: List[BrowningZone] = field(default_factory=list)


@dataclass
class TextureMetrics:
    grayscale_std: float
    laplacian_variance: float
    local_contrast: float
    texture_score: float  # 0 to 100
    disclaimer: str = (
        "Measures visible image surface texture. "
        "Does not determine physical softness, chewiness, or taste."
    )


@dataclass
class PerfectionScore:
    circularity_score: float  # 0 to 100
    ideal_circle_deviation_score: float  # 0 to 100
    radius_stability_score: float  # 0 to 100
    ellipse_roundness_score: float  # 0 to 100
    corner_quality_score: float  # 0 to 100
    symmetry_score: float  # 0 to 100
    boundary_smoothness_score: float  # 0 to 100
    center_accuracy_score: float  # 0 to 100
    browning_control_score: float  # 0 to 100
    texture_score: float  # 0 to 100
    raw_weighted_score: float  # Score before sanity gate
    perfection_index: float  # Final 0 to 100
    geometric_type: str
    verdict: str
    verdict_quote: str
    score_cap_applied: bool = False
    score_cap_reason: Optional[str] = None


@dataclass
class ChapatiAnalysisResult:
    id: str  # e.g. "CHAPATI-0001"
    timestamp: str
    image_filename: str
    geometry: GeometricMetrics
    radial: RadialMetrics
    browning: BrowningMetrics
    texture: TextureMetrics
    score: PerfectionScore
    # Forward-compatibility: Stage 2 Shape Genome representation
    stage2_genome_vector: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to clean JSON-serializable dictionary."""
        return asdict(self)
