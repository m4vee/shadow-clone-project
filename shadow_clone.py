"""
Shadow Clone Jutsu project - Phase 4
Make the Shadow Clone sign -> copies of you appear in the background.

Needs sign_detect.py in the same folder (it provides the sign detector).

Keys:  q = quit
       c = dismiss the clones
       d = toggle debug overlay (hand skeletons + detector numbers)
       [ / ] = lower / raise the segmentation threshold
"""

import time
import cv2
import numpy as np
import mediapipe as mp

from sign_detect import (
    analyze, to_px,
    HOLD_SECONDS, GRACE_SECONDS, SMOOTHING, SCORE_ON, COOLDOWN_SECONDS,
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_seg = mp.solutions.selfie_segmentation

WORK_W, WORK_H = 960, 540      # everything runs at this size for speed
FADE_IN = 0.5                  # seconds for clones to fade in
FADE_OUT = 0.4                 # seconds for clones to fade out

# Each clone: (x offset as fraction of frame width, scale, brightness)
# Listed back-to-front: outer/smaller/darker clones are drawn first.
CLONES = [
    (-0.42, 0.80, 0.70),
    (0.42, 0.80, 0.70),
    (-0.30, 0.88, 0.80),
    (0.30, 0.88, 0.80),
    (-0.18, 0.94, 0.90),
    (0.18, 0.94, 0.90),
]


def clean_mask(mask, threshold):
    """Turn the raw segmentation mask into a clean, soft-edged alpha (0..1)."""
    binary = (mask > threshold).astype(np.uint8)
    binary = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)

    # drop small floating specks, keep the big blob(s) that are you
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i, a in enumerate(areas) if a >= 0.15 * areas.max()]
        binary = np.isin(labels, keep).astype(np.uint8)

    return cv2.GaussianBlur(binary.astype(np.float32), (7, 7), 0)


def render(frame, alpha, strength):
    """Draw clones behind you, then draw the real you on top."""
    h, w = frame.shape[:2]
    out = frame.astype(np.float32)

    if strength > 0:
        for dx, s, bright in CLONES:
            # scale around the bottom-center of the frame, then shift sideways
            M = np.float32([[s, 0, (1 - s) * w / 2 + dx * w],
                            [0, s, (1 - s) * h]])
            warped = cv2.warpAffine(frame, M, (w, h)).astype(np.float32) * bright
            warped_a = cv2.warpAffine(alpha, M, (w, h))[..., None] * strength
            out = warped * warped_a + out * (1 - warped_a)

    # the real you, on top of every clone
    a = alpha[..., None]
    out = frame.astype(np.float32) * a + out * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold = 0.65
    debug_on = True

    # sign detection state
    smooth_score = 0.0
    hold_start = None
    last_seen = 0.0
    last_trigger = 0.0

    # clone state
    clones_active = False
    spawn_time = 0.0
    dismiss_time = None

    with mp_hands.Hands(
        max_num_hands=2, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as hands, mp_seg.SelfieSegmentation(model_selection=0) as seg:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WORK_W, WORK_H))
            now = time.time()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # ---------- sign detection ----------
            hres = hands.process(rgb)
            hands_px, hand_lms = [], []
            if hres.multi_hand_landmarks:
                for hl in hres.multi_hand_landmarks:
                    hand_lms.append(hl)
                    hands_px.append(to_px(hl, WORK_W, WORK_H))

            raw_score, debug = analyze(hands_px)
            smooth_score = SMOOTHING * raw_score + (1 - SMOOTHING) * smooth_score
            if smooth_score >= SCORE_ON:
                last_seen = now
                if hold_start is None:
                    hold_start = now
            elif hold_start is not None and now - last_seen > GRACE_SECONDS:
                hold_start = None
            held = (now - hold_start) if hold_start else 0.0

            if (held >= HOLD_SECONDS and not clones_active
                    and now - last_trigger > COOLDOWN_SECONDS):
                print("SHADOW CLONE JUTSU!")
                clones_active = True
                spawn_time = now
                dismiss_time = None
                last_trigger = now
                hold_start = None

            # ---------- clone strength (fade in / out) ----------
            strength = 0.0
            if clones_active:
                if dismiss_time is None:
                    strength = min(1.0, (now - spawn_time) / FADE_IN)
                else:
                    strength = 1.0 - (now - dismiss_time) / FADE_OUT
                    if strength <= 0:
                        clones_active, strength, dismiss_time = False, 0.0, None

            # ---------- segmentation + compositing ----------
            alpha = clean_mask(seg.process(rgb).segmentation_mask, threshold)
            out = render(frame, alpha, strength)

            # ---------- overlay ----------
            if debug_on:
                for hl in hand_lms:
                    mp_draw.draw_landmarks(out, hl, mp_hands.HAND_CONNECTIONS)
                bar = int(300 * min(held / HOLD_SECONDS, 1.0))
                cv2.rectangle(out, (20, 20), (320, 45), (60, 60, 60), 2)
                cv2.rectangle(out, (20, 20), (20 + bar, 45), (0, 200, 255), -1)
                cv2.putText(out, f"score {smooth_score:.2f}  thr {threshold:.2f}",
                            (335, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                state = "CLONES ON (c to dismiss)" if clones_active else "make the sign"
                cv2.putText(out, state, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Shadow Clone Jutsu - Phase 4", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and clones_active and dismiss_time is None:
                dismiss_time = now
            elif key == ord("d"):
                debug_on = not debug_on
            elif key == ord("["):
                threshold = max(0.1, threshold - 0.05)
            elif key == ord("]"):
                threshold = min(0.95, threshold + 0.05)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
