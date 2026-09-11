"""
Database and Export Module for Chapati Analyzer™.
Handles SQLite persistence in data/chapati.db, structured records,
ID generation (CHAPATI-XXXX), history retrieval, and CSV/JSON exports.
Guarantees clean connection handling and resource deallocation.
"""

import os
import sqlite3
import json
import csv
import io
from contextlib import closing
from typing import List, Dict, Any, Optional
from src.models import ChapatiAnalysisResult


class Database:
    """Manages SQLite storage and historical query exports for chapati analyses."""

    def __init__(self, db_path: str = "data/chapati.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Initializes tables and indexes if they do not exist."""
        with closing(self._get_connection()) as conn:
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        image_filename TEXT NOT NULL,
                        area REAL NOT NULL,
                        perimeter REAL NOT NULL,
                        circularity REAL NOT NULL,
                        width REAL NOT NULL,
                        height REAL NOT NULL,
                        aspect_ratio REAL NOT NULL,
                        equivalent_diameter REAL NOT NULL,
                        center_deviation REAL NOT NULL,
                        center_accuracy REAL NOT NULL,
                        mean_radius REAL NOT NULL,
                        radius_variation REAL NOT NULL,
                        radius_stability REAL NOT NULL,
                        symmetry REAL NOT NULL,
                        edge_smoothness REAL NOT NULL,
                        burn_ratio REAL NOT NULL,
                        burn_classification TEXT NOT NULL,
                        texture_score REAL NOT NULL,
                        perfection_score REAL NOT NULL,
                        verdict TEXT NOT NULL,
                        verdict_quote TEXT NOT NULL,
                        stage2_genome_json TEXT,
                        full_result_json TEXT NOT NULL
                    )
                """)

    def generate_next_id(self) -> str:
        """Generates the next sequential ID: CHAPATI-0001, CHAPATI-0002, etc."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM analyses")
            count = cursor.fetchone()[0]
            next_idx = count + 1
            return f"CHAPATI-{next_idx:04d}"

    def save_analysis(self, result: ChapatiAnalysisResult) -> str:
        """Saves a complete ChapatiAnalysisResult into SQLite."""
        full_json = json.dumps(result.to_dict())
        genome_json = json.dumps(result.stage2_genome_vector) if result.stage2_genome_vector else None

        with closing(self._get_connection()) as conn:
            with conn:
                conn.execute("""
                    INSERT INTO analyses (
                        id, timestamp, image_filename, area, perimeter, circularity,
                        width, height, aspect_ratio, equivalent_diameter,
                        center_deviation, center_accuracy, mean_radius, radius_variation,
                        radius_stability, symmetry, edge_smoothness, burn_ratio,
                        burn_classification, texture_score, perfection_score, verdict,
                        verdict_quote, stage2_genome_json, full_result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.id,
                    result.timestamp,
                    result.image_filename,
                    result.geometry.area,
                    result.geometry.perimeter,
                    result.geometry.circularity,
                    result.geometry.width,
                    result.geometry.height,
                    result.geometry.aspect_ratio,
                    result.geometry.equivalent_diameter,
                    result.geometry.center_deviation_px,
                    result.geometry.center_accuracy_score,
                    result.radial.mean_radius,
                    result.radial.coefficient_of_variation,
                    result.radial.radius_stability_score,
                    result.radial.symmetry_score,
                    result.radial.boundary_smoothness_score,
                    result.browning.burn_ratio,
                    result.browning.classification,
                    result.texture.texture_score,
                    result.score.perfection_index,
                    result.score.verdict,
                    result.score.verdict_quote,
                    genome_json,
                    full_json,
                ))

        return result.id

    def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves full analysis dictionary by ID."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("SELECT full_result_json FROM analyses WHERE id = ?", (analysis_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["full_result_json"])
            return None

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """
        Retrieves history table items ordered by timestamp DESC.
        Returns empty list if no records exist.
        """
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, circularity, burn_ratio, perfection_score, verdict, burn_classification
                FROM analyses
                ORDER BY rowid DESC
            """)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def export_csv(self) -> str:
        """Exports all recorded analyses into a CSV string."""
        with closing(self._get_connection()) as conn:
            cursor = conn.execute("""
                SELECT id, timestamp, area, perimeter, circularity, aspect_ratio,
                       equivalent_diameter, center_accuracy, radius_stability,
                       symmetry, edge_smoothness, burn_ratio, burn_classification,
                       texture_score, perfection_score, verdict
                FROM analyses
                ORDER BY rowid ASC
            """)
            rows = cursor.fetchall()

        output = io.StringIO()
        if not rows:
            return "No analyses recorded."

        writer = csv.writer(output)
        writer.writerow([
            "ID", "Timestamp", "Area (px^2)", "Perimeter (px)", "Circularity",
            "Aspect Ratio", "Equivalent Diameter (px)", "Center Accuracy",
            "Radius Stability", "Symmetry", "Edge Smoothness", "Burn Ratio",
            "Burn Classification", "Texture Score", "Perfection Score", "Verdict"
        ])
        for row in rows:
            writer.writerow(list(row))

        return output.getvalue()
