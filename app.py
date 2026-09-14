from flask import Flask, render_template, request, jsonify
from model import detect_objects

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    # Check if frame was received
    if "frame" not in request.files:
        return jsonify({"error": "No frame received"}), 400

    image_file = request.files["frame"]

    # Read image
    image_bytes = image_file.read()

    # Run YOLO detection
    annotated_image, counts = detect_objects(image_bytes)

    if annotated_image is None:
        return jsonify({"error": "Could not process image"}), 400

    # Convert image bytes to hexadecimal
    image_hex = annotated_image.hex()

    return jsonify({
        "image": image_hex,
        "counts": counts
    })


if __name__ == "__main__":
    app.run(debug=True)