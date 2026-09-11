"""
Synthetic geometric shape generator for unit testing the Chapati Analyzer™.
Generates mathematical shapes A through H for shape ranking calibration
and regression testing without external dependencies.
"""

from typing import Tuple
import cv2
import numpy as np


class SyntheticShapeGenerator:
    """Generates synthetic binary masks and contours for rigorous testing."""

    @staticmethod
    def create_perfect_circle(radius: int = 150, img_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Test A: Mathematically perfect circle."""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        cv2.circle(mask, center, radius, 255, -1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_slightly_irregular_circle(radius: int = 150, img_size: int = 400, wobble: float = 0.04) -> Tuple[np.ndarray, np.ndarray]:
        """Test B: Slightly irregular circle with tiny organic wobble."""
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        center = (img_size // 2, img_size // 2)
        r = radius * (1.0 + wobble * np.sin(3 * angles) + (wobble * 0.5) * np.cos(5 * angles))
        xs = center[0] + r * np.cos(angles)
        ys = center[1] + r * np.sin(angles)
        pts = np.stack([xs, ys], axis=-1).astype(np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_ellipse(axis_a: int = 160, axis_b: int = 120, img_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Test C: Ellipse with aspect ratio ~ 1.33."""
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        center = (img_size // 2, img_size // 2)
        cv2.ellipse(mask, center, (axis_a, axis_b), angle=25, startAngle=0, endAngle=360, color=255, thickness=-1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_rounded_triangle(radius: int = 150, img_size: int = 400, triangle_factor: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
        """Test D: Rounded triangle / samosa shape (strong 3-fold harmonic)."""
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        center = (img_size // 2, img_size // 2)
        # 3-fold periodicity gives 3 vertices and 3 rounded sides
        r = radius * (1.0 + triangle_factor * np.cos(3 * angles))
        xs = center[0] + r * np.cos(angles)
        ys = center[1] + r * np.sin(angles)
        pts = np.stack([xs, ys], axis=-1).astype(np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_triangle(radius: int = 150, img_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Test E: Sharp equilateral triangle."""
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3]) - np.pi / 2
        center = (img_size // 2, img_size // 2)
        xs = center[0] + radius * 1.15 * np.cos(angles)
        ys = center[1] + radius * 1.15 * np.sin(angles)
        pts = np.stack([xs, ys], axis=-1).astype(np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_pentagon(radius: int = 150, img_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Test F: Five-sided polygon (regular pentagon)."""
        angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) - np.pi / 2
        center = (img_size // 2, img_size // 2)
        xs = center[0] + radius * np.cos(angles)
        ys = center[1] + radius * np.sin(angles)
        pts = np.stack([xs, ys], axis=-1).astype(np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_irregular_polygon(img_size: int = 400) -> Tuple[np.ndarray, np.ndarray]:
        """Test G: Highly irregular / asymmetric shape."""
        pts = np.array([
            [100, 70],
            [290, 110],
            [350, 220],
            [300, 330],
            [160, 350],
            [70, 240],
            [130, 160]
        ], dtype=np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_lobed_shape(radius: int = 140, img_size: int = 400, lobes: int = 4, lobe_depth: float = 0.16) -> Tuple[np.ndarray, np.ndarray]:
        """Test H: 4-lobed clover / flower shape."""
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        center = (img_size // 2, img_size // 2)
        r = radius * (1.0 + lobe_depth * np.cos(lobes * angles))
        xs = center[0] + r * np.cos(angles)
        ys = center[1] + r * np.sin(angles)
        pts = np.stack([xs, ys], axis=-1).astype(np.int32)
        mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return mask, contours[0]

    @staticmethod
    def create_synthetic_chapati_image(
        radius: int = 160, img_size: int = 500, with_browning: bool = True
    ) -> np.ndarray:
        """Realistic synthetic chapati image with dough tone and browning spots."""
        img = np.full((img_size, img_size, 3), (40, 45, 52), dtype=np.uint8)
        angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
        center = (img_size // 2, img_size // 2)
        r = radius * (1.0 + 0.02 * np.sin(4 * angles) + 0.015 * np.cos(7 * angles))
        xs = center[0] + r * np.cos(angles)
        ys = center[1] + r * np.sin(angles)
        poly_pts = np.stack([xs, ys], axis=-1).astype(np.int32)

        dough_mask = np.zeros((img_size, img_size), dtype=np.uint8)
        cv2.fillPoly(dough_mask, [poly_pts], 255)

        noise = np.random.normal(0, 5, (img_size, img_size)).astype(np.float32)
        dough_b = np.clip(140 + noise, 0, 255).astype(np.uint8)
        dough_g = np.clip(195 + noise, 0, 255).astype(np.uint8)
        dough_r = np.clip(225 + noise, 0, 255).astype(np.uint8)
        dough_bgr = cv2.merge([dough_b, dough_g, dough_r])

        img[dough_mask > 0] = dough_bgr[dough_mask > 0]

        if with_browning:
            np.random.seed(42)
            for _ in range(16):
                spot_angle = np.random.uniform(0, 2 * np.pi)
                spot_dist = np.random.uniform(20, radius * 0.8)
                sx = int(center[0] + spot_dist * np.cos(spot_angle))
                sy = int(center[1] + spot_dist * np.sin(spot_angle))
                spot_r = int(np.random.uniform(6, 18))
                cv2.circle(img, (sx, sy), spot_r, (45, 75, 110), -1)

            img = cv2.GaussianBlur(img, (3, 3), 0)

        return img
