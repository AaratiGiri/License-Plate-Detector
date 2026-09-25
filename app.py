from pathlib import Path
from datetime import datetime

import os
import cv2

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    Response,
    jsonify,
    send_from_directory,
    send_file,
    flash
)

from werkzeug.utils import secure_filename

from detection import (
    process_frame,
    get_recent_records,
    DETECTED_DIR,
    CSV_PATH
)


# ---------------------------------------------------------
# FLASK SETUP
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"

UPLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)


app = Flask(__name__)

app.secret_key = "license-plate-project-secret"

app.config["MAX_CONTENT_LENGTH"] = (
    500 * 1024 * 1024
)


# ---------------------------------------------------------
# ALLOWED FILE TYPES
# ---------------------------------------------------------

IMAGE_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
    "mkv",
    "mpeg",
    "mpg",
    "webm"
}


def get_extension(filename):

    if "." not in filename:
        return ""

    return filename.rsplit(".", 1)[1].lower()


def is_image(filename):

    return (
        get_extension(filename)
        in IMAGE_EXTENSIONS
    )


def is_video(filename):

    return (
        get_extension(filename)
        in VIDEO_EXTENSIONS
    )


# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------

@app.route("/")
def index():

    video_file = request.args.get(
        "video"
    )

    image_file = request.args.get(
        "image"
    )

    records = get_recent_records(
        limit=20
    )

    return render_template(
        "index.html",
        video_file=video_file,
        image_file=image_file,
        records=records
    )


# ---------------------------------------------------------
# IMAGE OR VIDEO UPLOAD
# ---------------------------------------------------------

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_file():

    if "media" not in request.files:

        flash(
            "No file selected."
        )

        return redirect(
            url_for("index")
        )


    file = request.files["media"]


    if file.filename == "":

        flash(
            "Please select an image or video."
        )

        return redirect(
            url_for("index")
        )


    if not (
        is_image(file.filename)
        or is_video(file.filename)
    ):

        flash(
            "Unsupported file type."
        )

        return redirect(
            url_for("index")
        )


    safe_name = secure_filename(
        file.filename
    )


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    filename = (
        f"{timestamp}_{safe_name}"
    )


    save_path = (
        UPLOAD_DIR /
        filename
    )


    file.save(
        str(save_path)
    )


    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    if is_image(filename):

        image = cv2.imread(
            str(save_path)
        )

        if image is None:

            flash(
                "Could not read the uploaded image."
            )

            return redirect(
                url_for("index")
            )


        annotated_frame, detections = process_frame(
            frame=image,
            source=f"image:{filename}",
            save_detections=True,
            confidence_threshold=0.25,
            image_size=960
        )


        result_name = (
            f"result_{filename}"
        )


        result_path = (
            UPLOAD_DIR /
            result_name
        )


        cv2.imwrite(
            str(result_path),
            annotated_frame
        )


        return redirect(
            url_for(
                "index",
                image=result_name
            )
        )


    # -----------------------------------------------------
    # VIDEO
    # -----------------------------------------------------

    return redirect(
        url_for(
            "index",
            video=filename
        )
    )


# ---------------------------------------------------------
# SERVE UPLOADED/RESULT IMAGE
# ---------------------------------------------------------

@app.route(
    "/uploads/<filename>"
)
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_DIR,
        filename
    )


# ---------------------------------------------------------
# CONVERT FRAME TO MJPEG
# ---------------------------------------------------------

def encode_frame(frame):

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            85
        ]
    )


    if not success:
        return None


    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + buffer.tobytes()
        + b"\r\n"
    )


# ---------------------------------------------------------
# PROCESS UPLOADED VIDEO
# ---------------------------------------------------------

def generate_video_frames(video_path):

    cap = cv2.VideoCapture(
        str(video_path)
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    if fps <= 1:
        fps = 25


    # Save approximately once per second
    save_every = max(
        int(fps),
        1
    )


    frame_number = 0


    while True:

        success, frame = cap.read()


        if not success:
            break


        should_save = (
            frame_number %
            save_every == 0
        )


        annotated_frame, detections = process_frame(
            frame=frame,
            source=f"video:{video_path.name}",
            save_detections=should_save,
            confidence_threshold=0.25,
            image_size=960
        )


        frame_number += 1


        encoded = encode_frame(
            annotated_frame
        )


        if encoded is not None:
            yield encoded


    cap.release()


# ---------------------------------------------------------
# VIDEO STREAM
# ---------------------------------------------------------

@app.route(
    "/video_feed/<filename>"
)
def video_feed(filename):

    filename = secure_filename(
        filename
    )


    video_path = (
        UPLOAD_DIR /
        filename
    )


    if not video_path.exists():

        return (
            jsonify({"error": "Video not found"}),
            404
        )


    return Response(
        generate_video_frames(
            video_path
        ),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ---------------------------------------------------------
# LIVE WEBCAM
# ---------------------------------------------------------

def generate_webcam_frames():

    if os.name == "nt":

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

    else:

        cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        print(
            "Could not open webcam."
        )

        return


    frame_number = 0


    while True:

        success, frame = cap.read()


        if not success:
            break


        should_save = (
            frame_number % 30 == 0
        )


        annotated_frame, detections = process_frame(
            frame=frame,
            source="webcam",
            save_detections=should_save,
            confidence_threshold=0.25,
            image_size=640
        )


        frame_number += 1


        encoded = encode_frame(
            annotated_frame
        )


        if encoded is not None:
            yield encoded


    cap.release()


@app.route("/webcam_feed")
def webcam_feed():

    return Response(
        generate_webcam_frames(),
        mimetype=(
            "multipart/x-mixed-replace;"
            " boundary=frame"
        )
    )


# ---------------------------------------------------------
# DETECTED PLATE IMAGE
# ---------------------------------------------------------

@app.route(
    "/detected/<filename>"
)
def detected_plate(filename):

    return send_from_directory(
        DETECTED_DIR,
        filename
    )


# ---------------------------------------------------------
# API HISTORY
# ---------------------------------------------------------

@app.route("/api/records")
def api_records():

    limit_param = request.args.get("limit", "30")

    if limit_param == "all":
        records = get_recent_records(limit=999999)
    else:
        try:
            limit = int(limit_param)
        except ValueError:
            limit = 30
        records = get_recent_records(limit=limit)

    return jsonify(records)


# ---------------------------------------------------------
# API STATS
# ---------------------------------------------------------

@app.route("/api/stats")
def api_stats():

    import csv as _csv

    total = 0
    plates_read = 0
    last_source = "-"

    try:
        with open(CSV_PATH, "r", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)

        total = len(rows)

        plates_read = sum(
            1 for r in rows
            if r.get("plate_text", "").strip()
        )

        if rows:
            src = rows[-1].get("source", "")
            if "webcam" in src:
                last_source = "Webcam"
            elif "video" in src:
                last_source = "Video"
            elif "image" in src:
                last_source = "Image"
            else:
                last_source = src

    except Exception:
        pass

    return jsonify({
        "total": total,
        "plates": plates_read,
        "last_source": last_source
    })


# ---------------------------------------------------------
# DOWNLOAD CSV
# ---------------------------------------------------------

@app.route("/download_csv")
def download_csv():

    return send_file(
        CSV_PATH,
        as_attachment=True,
        download_name="detections.csv"
    )


# ---------------------------------------------------------
# CLEAR HISTORY
# ---------------------------------------------------------

@app.route("/clear_history", methods=["POST"])
def clear_history():

    import shutil

    # Wipe CSV (keep header only)
    try:
        import csv as _csv
        with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
            writer = _csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "source",
                    "confidence",
                    "plate_text",
                    "plate_image"
                ]
            )
            writer.writeheader()
    except Exception as e:
        print("Could not clear CSV:", e)

    # Delete all saved plate images
    try:
        for img_file in DETECTED_DIR.iterdir():
            if img_file.is_file():
                img_file.unlink()
    except Exception as e:
        print("Could not clear detected plates:", e)

    return redirect(url_for("index"))



# ---------------------------------------------------------
# RUN APP
# ---------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False
    )