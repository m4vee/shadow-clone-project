"""
Shadow Clone Jutsu project - Phase 3
Cuts you out of the webcam feed with MediaPipe Selfie Segmentation and shows
the cutout on a checkerboard next to the original.

Keys:  q = quit
       [ = lower mask threshold (cutout includes more)
       ] = raise mask threshold (cutout is stricter)
"""

import cv2
import numpy as np
import mediapipe as mp

mp_seg = mp.solutions.selfie_segmentation

threshold = 0.5   # mask value above this counts as "person"


def checkerboard(h, w, size=40):
    ys, xs = np.indices((h, w))
    board = ((ys // size + xs // size) % 2) * 40 + 60
    return np.dstack([board] * 3).astype(np.uint8)


def make_cutout(frame, mask, threshold):
    """Return (cutout_bgr, alpha) where alpha is 0..1 float, soft at the edges."""
    alpha = (mask > threshold).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (9, 9), 0)
    return frame, alpha[..., None]


def main():
    global threshold

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    bg = None

    with mp_seg.SelfieSegmentation(model_selection=0) as seg:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if bg is None:
                bg = checkerboard(h, w)

            results = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mask = results.segmentation_mask

            cutout, alpha = make_cutout(frame, mask, threshold)
            composite = (cutout * alpha + bg * (1 - alpha)).astype(np.uint8)

            # show original and cutout side by side, scaled to fit the screen
            left = cv2.resize(frame, (640, 360))
            right = cv2.resize(composite, (640, 360))
            combined = np.hstack([left, right])
            cv2.putText(combined, f"threshold {threshold:.2f}  ([ and ] to adjust)",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Shadow Clone Jutsu - Phase 3", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("["):
                threshold = max(0.1, threshold - 0.05)
            elif key == ord("]"):
                threshold = min(0.95, threshold + 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
