import base64

from flask import Flask, request, jsonify, abort, send_from_directory
from flask_cors import CORS
from skimage.io import imread
from src.analysis.roi_pipeline import process_roi_from_mip
from src.io.sholl import sholl_analysis
import os
import json
import glob
import csv

app = Flask(__name__)
CORS(app)

OUTPUT_DIR = "outputs"

@app.route("/roi", methods=["POST"])
def process_roi():
    data = request.json
    image_path = data.get("imagePath")
    if not os.path.exists(image_path):
        return jsonify({"error": "Imaginea nu există"}), 404

    process_roi_from_mip(image_path, OUTPUT_DIR)

    binary_path = os.path.join(OUTPUT_DIR, "roi_binary.tif")
    if not os.path.exists(binary_path):
        return jsonify({"error": "ROI binarizat lipsă"}), 500

    binary_img = imread(binary_path)
    sholl_analysis(binary_img, step_size=5, max_radius=250, save_path=os.path.join(OUTPUT_DIR, "sholl_roi_binary.png"))

    return jsonify({"message": "ROI și Sholl analizate"}), 200


@app.route("/batch", methods=["POST"])
def batch_pipeline():
    data = request.json
    folder = data.get("path")
    if not os.path.exists(folder):
        return jsonify({"error": "Folderul nu există"}), 404

    results = []

    for tif_path in glob.glob(os.path.join(folder, "*.tif")):
        name = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"🔁 Procesăm: {name}")
        process_roi_from_mip(tif_path, OUTPUT_DIR)

        binary_path = os.path.join(OUTPUT_DIR, "roi_binary.tif")
        if os.path.exists(binary_path):
            binary_img = imread(binary_path)
            save_path = os.path.join(OUTPUT_DIR, f"sholl_{name}.png")
            sholl_analysis(binary_img, step_size=5, max_radius=250, save_path=save_path)
            results.append(name)

    return jsonify({"processed": results})


@app.route("/features", methods=["GET"])
def get_features():
    csv_path = os.path.join(OUTPUT_DIR, "features.csv")
    if not os.path.exists(csv_path):
        return jsonify([])

    features = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(row)

    return jsonify(features)


@app.route("/sholl/<neuron_id>", methods=["GET"])
def get_sholl(neuron_id):
    json_path = os.path.join(OUTPUT_DIR, "sholl_data.json")
    if not os.path.exists(json_path):
        return jsonify({})

    with open(json_path) as f:
        data = json.load(f)
        return jsonify(data.get(neuron_id, []))

@app.route("/", methods=["GET"])
def index():
    return "Hello, server works"

@app.route("/images", methods=["GET"])
def get_all_images():
    images_by_roi = {}
    full_rgb_image = None

    for root, dirs, files in os.walk(OUTPUT_DIR):
        png_files = [f for f in files if f.endswith(".png")]
        if png_files:
            roi_name = os.path.relpath(root, OUTPUT_DIR)

            for filename in png_files:
                file_path = os.path.join(root, filename)

                with open(file_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                    image_data = {
                        "filename": filename,
                        "description": f"Reprezintă rezultatul pentru ROI: {roi_name}",
                        "data": f"data:image/png;base64,{encoded}"
                    }

                    # dacă este full_rgb.png și e în directorul principal
                    if filename == "full_rgb.png" and roi_name == ".":
                        full_rgb_image = {
                            "filename": filename,
                            "description": "Imagine RGB completă",
                            "data": f"data:image/png;base64,{encoded}"
                        }
                    else:
                        if roi_name not in images_by_roi:
                            images_by_roi[roi_name] = []
                        images_by_roi[roi_name].append(image_data)

    return jsonify({
        "full_rgb": full_rgb_image,
        "images_by_roi": images_by_roi
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
