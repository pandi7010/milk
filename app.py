from flask import Flask, render_template, request, Response
import cv2
import os
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIG ----------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load YOLOv5 model
model = YOLO("yolov5n.pt")   # make sure yolov5n.pt is in same folder
CONFIDENCE = 0.4
# ---------------------------------------


def detect_and_count(frame):
    """
    Run YOLO inference on a frame, draw bounding boxes,
    and return processed frame + object count
    """
    results = model(frame, conf=CONFIDENCE, verbose=False)
    res = results[0]

    count = 0

    if res.boxes is not None:
        boxes = res.boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box[:4])
            count += 1

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

    cv2.putText(
        frame,
        f"Count: {count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        3
    )

    return frame, count


# ---------------- ROUTES ----------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/image", methods=["POST"])
def image():
    file = request.files["file"]
    if file.filename == "":
        return "No file uploaded"

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return "Invalid image"

    img, count = detect_and_count(img)
    cv2.imwrite(path, img)

    return f"Milk Packets Counted: {count}"


@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame, _ = detect_and_count(frame)
        ret, buffer = cv2.imencode(".jpg", frame)

        frame = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame +
            b"\r\n"
        )

    cap.release()


# ---------------- MAIN ------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
