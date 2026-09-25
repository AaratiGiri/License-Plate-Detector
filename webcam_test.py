import cv2

from detection import detector


# ============================================================
# OPEN WEBCAM
# ============================================================

camera = cv2.VideoCapture(0)


if not camera.isOpened():

    print("Cannot open webcam")

    raise SystemExit


print("Webcam started.")

print("Press Q to stop.")


# ============================================================
# LIVE DETECTION
# ============================================================

while True:

    success, frame = camera.read()


    if not success:

        print("Cannot read webcam frame")

        break


    annotated_frame, detections = (
        detector.process_frame(
            frame,
            source="webcam_test",
            save_record=False
        )
    )


    cv2.imshow(
        "YOLO License Plate Detection",
        annotated_frame
    )


    if (
        cv2.waitKey(1)
        & 0xFF
        == ord("q")
    ):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()