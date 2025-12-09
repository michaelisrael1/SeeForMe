from flask import Flask, request, jsonify
from PIL import Image
from ultralytics import YOLO
import io

model = YOLO('yolov5s.pt')
model.cuda()

app = Flask(__name__)


@app.route("/detect/", methods=["POST"])
def detect_image():
    """
    Handles image uploads, runs YOLOv5 detection, and returns JSON results.
    The client must send the image as a multipart/form-data with a field named 'file'.
    """
    if model is None:
        return jsonify({"error": "YOLOv5 model is not loaded."}), 503

    # Flask receives uploaded files in request.files (for multipart/form-data)
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' field found in the upload request."}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected."}), 400

    try:
        # Read the image file stream
        image_bytes = file.read()

        # Convert bytes to a PIL Image object
        img = Image.open(io.BytesIO(image_bytes))

        # Run YOLOv5 inference
        results = model(img)

        detections = []
        for result in results:
            detections = [result.names[cls.item()]
                          for cls in result.boxes.cls.int()]

        return jsonify({"detected_objects": detections})

    except Exception as e:
        print(f"An unexpected error occurred during detection: {e}")
        return jsonify({"error": f"Internal server error during processing: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
