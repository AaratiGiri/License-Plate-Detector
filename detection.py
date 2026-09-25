from pathlib import Path
from datetime import datetime

import csv
import re
import threading

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model" / "best.pt"

DETECTED_DIR = BASE_DIR / "detected_plates"

CSV_PATH = BASE_DIR / "detections.csv"


# Create folder automatically if it does not exist
DETECTED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD TRAINED YOLO MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nTrained model not found:\n{MODEL_PATH}\n"
        "Place best.pt inside the model folder."
    )


print("Loading trained YOLO model...")

model = YOLO(str(MODEL_PATH))

print("YOLO model loaded successfully ✅")


# ============================================================
# LOAD EASYOCR
# ============================================================

ocr_reader = None

try:
    import easyocr

    print("Loading EasyOCR...")

    ocr_reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("EasyOCR loaded successfully ✅")

except Exception as error:

    print(
        "EasyOCR could not be loaded:",
        error
    )

    print(
        "YOLO detection will still work."
    )


# ============================================================
# CSV SETTINGS
# ============================================================

CSV_FIELDS = [
    "timestamp",
    "source",
    "confidence",
    "plate_text",
    "plate_image"
]

csv_lock = threading.Lock()


def initialize_csv():

    if (
        not CSV_PATH.exists()
        or CSV_PATH.stat().st_size == 0
    ):

        with open(
            CSV_PATH,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS
            )

            writer.writeheader()


initialize_csv()


# ============================================================
# OCR FUNCTION
# ============================================================

def read_plate_text(plate_image):

    if ocr_reader is None:
        return ""

    if (
        plate_image is None
        or plate_image.size == 0
    ):
        return ""

    try:

        results = ocr_reader.readtext(
            plate_image,
            detail=0,
            paragraph=True
        )

        if not results:
            return ""

        text = " ".join(results)

        # Remove repeated spaces
        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        return text

    except Exception as error:

        print("OCR error:", error)

        return ""


# ============================================================
# SAVE DETECTION INTO CSV
# ============================================================

def save_detection_record(
    timestamp,
    source,
    confidence,
    plate_text,
    plate_image
):

    row = {
        "timestamp": timestamp,
        "source": source,
        "confidence": f"{confidence:.4f}",
        "plate_text": plate_text,
        "plate_image": plate_image
    }

    with csv_lock:

        with open(
            CSV_PATH,
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=CSV_FIELDS
            )

            writer.writerow(row)


# ============================================================
# MAIN DETECTION FUNCTION
# ============================================================

def process_frame(
    frame,
    source="unknown",
    save_detections=False,
    confidence_threshold=0.25,
    image_size=640
):

    """
    Detect license plates from one image/video/webcam frame.

    Returns:
        annotated_frame
        detections
    """

    if frame is None or frame.size == 0:

        return frame, []


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=confidence_threshold,
        imgsz=image_size,
        verbose=False
    )


    result = results[0]


    # YOLO automatically draws bounding boxes
    annotated_frame = result.plot()


    detections = []


    if result.boxes is None:

        return (
            annotated_frame,
            detections
        )


    image_height, image_width = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # PROCESS EVERY DETECTED PLATE
    # --------------------------------------------------------

    for index, box in enumerate(
        result.boxes
    ):

        coordinates = (
            box.xyxy[0]
            .cpu()
            .tolist()
        )


        x1, y1, x2, y2 = coordinates


        # Convert to integers
        x1 = max(
            0,
            int(x1)
        )

        y1 = max(
            0,
            int(y1)
        )

        x2 = min(
            image_width,
            int(x2)
        )

        y2 = min(
            image_height,
            int(y2)
        )


        confidence = float(
            box.conf[0].item()
        )


        # Invalid box protection
        if (
            x2 <= x1
            or y2 <= y1
        ):
            continue


        # ----------------------------------------------------
        # CROP LICENSE PLATE  (with padding)
        # ----------------------------------------------------

        PAD = 6   # pixels of padding around the plate box

        px1 = max(0, x1 - PAD)
        py1 = max(0, y1 - PAD)
        px2 = min(image_width,  x2 + PAD)
        py2 = min(image_height, y2 + PAD)

        plate_crop = frame[
            py1:py2,
            px1:px2
        ].copy()


        plate_text = ""

        plate_filename = ""


        # ----------------------------------------------------
        # ENHANCE PLATE IMAGE
        # ----------------------------------------------------

        def enhance_plate(crop):
            """Upscale, sharpen and boost contrast of plate crop."""

            import numpy as np

            if crop is None or crop.size == 0:
                return crop

            h, w = crop.shape[:2]

            # --- Upscale so the plate is at least 400 x 120 px ---
            target_w = max(w, 400)
            target_h = max(h, 120)

            # Keep aspect ratio
            scale = max(target_w / w, target_h / h)

            new_w = int(w * scale)
            new_h = int(h * scale)

            # Use LANCZOS (high quality) for upscaling
            upscaled = cv2.resize(
                crop,
                (new_w, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )

            # --- Sharpening kernel ---
            sharpen_kernel = np.array([
                [ 0, -1,  0],
                [-1,  5, -1],
                [ 0, -1,  0]
            ], dtype=np.float32)

            sharpened = cv2.filter2D(
                upscaled,
                -1,
                sharpen_kernel
            )

            # --- Contrast enhancement (CLAHE on L channel) ---
            lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)

            l_channel, a, b = cv2.split(lab)

            clahe = cv2.createCLAHE(
                clipLimit=2.5,
                tileGridSize=(4, 4)
            )

            l_channel = clahe.apply(l_channel)

            enhanced = cv2.merge([l_channel, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            return enhanced


        # ----------------------------------------------------
        # SAVE DETECTION
        # ----------------------------------------------------

        if save_detections:

            unique_time = (
                datetime.now()
                .strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )


            plate_filename = (
                f"plate_"
                f"{unique_time}_"
                f"{index}.jpg"
            )


            plate_path = (
                DETECTED_DIR
                / plate_filename
            )


            # Enhance then save
            enhanced_plate = enhance_plate(plate_crop)

            cv2.imwrite(
                str(plate_path),
                enhanced_plate,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )


            # Run OCR on enhanced image for better accuracy
            plate_text = read_plate_text(
                enhanced_plate
            )


            timestamp = (
                datetime.now()
                .strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )


            # Save into CSV
            save_detection_record(
                timestamp=timestamp,
                source=source,
                confidence=confidence,
                plate_text=plate_text,
                plate_image=plate_filename
            )



        detections.append(
            {
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ],

                "confidence":
                    confidence,

                "plate_text":
                    plate_text,

                "plate_image":
                    plate_filename
            }
        )


    return (
        annotated_frame,
        detections
    )


# ============================================================
# READ DETECTION HISTORY
# ============================================================

def get_recent_records(
    limit=20
):

    initialize_csv()


    with csv_lock:

        with open(
            CSV_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            reader = csv.DictReader(
                file
            )

            records = list(
                reader
            )


    # Newest records first
    records.reverse()


    return records[:limit]