"""
CHAPATI ANALYZER™ — Web Application Server
Flask-based backend providing REST API and serving Progressive Web App (PWA) assets.
"""

import os
import base64
import io
from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    send_from_directory,
    Response,
    send_file,
)
from src.pipeline import ChapatiPipeline
from src.preprocessing import ImageValidationError
from src.segmentation import ChapatiDetectionError
from src.database import Database

app = Flask(
    __name__,
    static_folder="static",
    template_folder="templates"
)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB max upload

pipeline = ChapatiPipeline(data_dir="data")
db = Database("data/chapati.db")


# PWA Static Routing
@app.route("/manifest.json")
def manifest():
    return send_from_directory("static", "manifest.json", mimetype="application/manifest+json")


@app.route("/sw.js")
def service_worker():
    response = send_from_directory("static", "sw.js", mimetype="application/javascript")
    response.headers["Service-Worker-Allowed"] = "/"
    response.headers["Cache-Control"] = "no-cache"
    return response


@app.route("/data/images/<path:filename>")
def serve_analyzed_image(filename):
    images_dir = os.path.join(os.path.abspath("data"), "images")
    return send_from_directory(images_dir, filename)


# Main App Route
@app.route("/")
def index():
    return render_template("index.html")


# API Routes
@app.route("/api/analyze", methods=["POST"])
def analyze_chapati():
    """
    Accepts multipart file upload or JSON payload with base64 image data.
    Runs computer vision pipeline and returns structured laboratory analysis.
    """
    image_bytes = None
    filename_hint = "upload.jpg"

    # Check for multipart file upload
    if "image" in request.files:
        file = request.files["image"]
        if file.filename != "":
            filename_hint = file.filename
            image_bytes = file.read()

    # Check for JSON base64 upload (useful for camera captures from frontend canvas)
    elif request.is_json:
        data = request.get_json()
        if data and "image_data" in data:
            raw_data = data["image_data"]
            # Strip data URL prefix if present: "data:image/jpeg;base64,..."
            if "," in raw_data:
                raw_data = raw_data.split(",", 1)[1]
            try:
                image_bytes = base64.b64decode(raw_data)
                filename_hint = data.get("filename", "camera_capture.jpg")
            except Exception:
                return jsonify({
                    "success": False,
                    "error": "Invalid base64 image data payload.",
                    "code": "INVALID_PAYLOAD"
                }), 400

    if not image_bytes:
        return jsonify({
            "success": False,
            "error": "No image payload received. Please upload an image or capture a photo.",
            "code": "EMPTY_PAYLOAD"
        }), 400

    try:
        result = pipeline.process_image_bytes(image_bytes, filename_hint=filename_hint)
        return jsonify({
            "success": True,
            "data": result
        }), 200

    except ImageValidationError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "code": "IMAGE_VALIDATION_ERROR"
        }), 400

    except ChapatiDetectionError as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "code": "DETECTION_FAILURE"
        }), 422

    except Exception as e:
        # Graceful handling: never expose raw stack traces to normal users
        app.logger.exception("Unexpected error during chapati analysis")
        return jsonify({
            "success": False,
            "error": "An unexpected optical computation anomaly occurred. Please try another image.",
            "code": "PIPELINE_ERROR"
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """Returns list of past analyses."""
    summaries = db.get_all_summaries()
    return jsonify({
        "success": True,
        "count": len(summaries),
        "data": summaries
    })


@app.route("/api/analysis/<analysis_id>", methods=["GET"])
def get_analysis_details(analysis_id):
    """Retrieves full analysis for a single specimen ID."""
    record = db.get_analysis_by_id(analysis_id)
    if not record:
        return jsonify({
            "success": False,
            "error": f"Specimen record {analysis_id} not found."
        }), 404

    return jsonify({
        "success": True,
        "data": record
    })


@app.route("/api/export/csv", methods=["GET"])
def export_csv():
    """Generates and downloads CSV of historical records."""
    csv_data = db.export_csv()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=chapati_analyses.csv"}
    )


@app.route("/api/export/json/<analysis_id>", methods=["GET"])
def export_json(analysis_id):
    """Exports structured JSON for a single analysis."""
    record = db.get_analysis_by_id(analysis_id)
    if not record:
        return jsonify({"error": "Record not found"}), 404

    # Format JSON clean output as requested in PRD Section 27 and Section 30
    clean_export = {
        "id": record["id"],
        "timestamp": record["timestamp"],
        "geometry": {
            "area": round(record["geometry"]["area"], 1),
            "perimeter": round(record["geometry"]["perimeter"], 1),
            "circularity": round(record["geometry"]["circularity"], 4),
            "aspect_ratio": round(record["geometry"]["aspect_ratio"], 3),
            "diameter": round(record["geometry"]["equivalent_diameter"], 1),
            "solidity": round(record["geometry"].get("solidity", 1.0), 4),
            "ellipse_axis_ratio": round(record["geometry"].get("ellipse_axis_ratio", 1.0), 3),
            "eccentricity": round(record["geometry"].get("ellipse_eccentricity", 0.0), 3),
        },
        "shape": {
            "geometric_type": record["radial"].get("geometric_type", "ROUND / IRREGULAR"),
            "ideal_circle_deviation": round(record["radial"].get("ideal_circle_deviation_score", 100.0), 1),
            "radial_rms_error": round(record["radial"].get("ideal_circle_rms_error", 0.0) * 100.0, 2),
            "radius_stability": round(record["radial"]["radius_stability_score"], 1),
            "symmetry": round(record["radial"]["symmetry_score"], 1),
            "boundary_smoothness": round(record["radial"].get("boundary_smoothness_score", record["radial"].get("edge_smoothness_score", 100.0)), 1),
            "corner_strength": record["radial"].get("corner_strength", "LOW"),
            "harmonic_irregularity": round(record["radial"].get("harmonic_irregularity", 0.0), 2),
            "center_accuracy": round(record["geometry"]["center_accuracy_score"], 1)
        },
        "browning": {
            "burn_ratio": round(record["browning"]["burn_ratio"], 4),
            "classification": record["browning"]["classification"]
        },
        "texture": {
            "score": round(record["texture"]["texture_score"], 1)
        },
        "score": {
            "perfection_index": record["score"]["perfection_index"],
            "geometric_type": record["score"].get("geometric_type", record["radial"].get("geometric_type")),
            "verdict": record["score"]["verdict"],
            "quote": record["score"]["verdict_quote"],
            "score_cap_applied": record["score"].get("score_cap_applied", False)
        }
    }

    return jsonify(clean_export)


if __name__ == "__main__":
    # Host on all interfaces so phone/mobile devices on local LAN can connect
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
