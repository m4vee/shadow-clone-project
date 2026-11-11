"""
Shadow Clone Jutsu project - Phase 2
Rule-based detection of the Shadow Clone hand sign (crossed index + middle
fingers of both hands), with smoothing and a hold timer that tolerates flicker.

Keys:  q = quit
Tune the THRESHOLDS below using the debug numbers shown on screen.
"""

import time
import cv2
import numpy as np
import mediapipe as mp

# ---------------- THRESHOLDS (tune these) ----------------
PALM_CLOSE = 1.9      # palm centers closer than this many "hand sizes"
TIPS_CLOSE = 0.8      # index/middle fingertips of the two hands this close
UP_MARGIN = 0.05      # fingertip must be this many hand sizes above the wrist
MAX_ANGLE = 150       # max angle (degrees) between the two hands' finger directions
EXT_RATIO = 1.5       # finger counts as "extended" if tip is this much farther from wrist than its base
FINGER_MIN = 6        # out of 8 (4 fingers x 2 hands) must match the sign's finger shape
HOLD_SECONDS = 0.5    # how long the sign must be held to trigger
GRACE_SECONDS = 0.3   # flicker tolerance: dropouts shorter than this are ignored
SMOOTHING = 0.4       # 0..1, higher = reacts faster, lower = steadier
SCORE_ON = 0.6        # smoothed score needed to count as "sign present"
COOLDOWN_SECONDS = 2.0
# ----------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

WRIST, INDEX_TIP, MIDDLE_TIP = 0, 8, 12
PALM_POINTS = [0, 5, 9, 13, 17]


def to_px(hand, w, h):
    return np.array([[lm.x * w, lm.y * h] for lm in hand.landmark])


def hand_size(pts):
    # wrist -> middle finger base, a stable scale reference
    return np.linalg.norm(pts[0] - pts[9]) + 1e-6


def palm_center(pts):
    return pts[PALM_POINTS].mean(axis=0)


# (tip, base, should_be_extended): index + middle out, ring + pinky curled
FINGER_PATTERN = [(8, 5, True), (12, 9, True), (16, 13, False), (20, 17, False)]


def finger_matches(pts):
    """How many of the 4 fingers match the sign's hand shape (0..4)."""
    n = 0
    for tip, base, want_extended in FINGER_PATTERN:
        ratio = np.linalg.norm(pts[tip] - pts[0]) / (np.linalg.norm(pts[base] - pts[0]) + 1e-6)
        if (ratio > EXT_RATIO) == want_extended:
            n += 1
    return n


def analyze(hands_px):
    """Return (score 0..1, debug dict) for the current frame."""
    if len(hands_px) < 2:
        return 0.0, {"hands": len(hands_px)}

    # sort left-to-right on screen so "a" is always the left hand and "b" the
    # right hand (MediaPipe's order can swap between frames)
    a, b = sorted(hands_px[:2], key=lambda p: palm_center(p)[0])
    size = (hand_size(a) + hand_size(b)) / 2

    palm_d = np.linalg.norm(palm_center(a) - palm_center(b)) / size

    tips_a = a[[INDEX_TIP, MIDDLE_TIP]]
    tips_b = b[[INDEX_TIP, MIDDLE_TIP]]
    tip_d = min(
        np.linalg.norm(p - q) for p in tips_a for q in tips_b
    ) / size

    # fingers pointing up: tip is above wrist (smaller y) by a margin
    up_a = (a[0][1] - a[INDEX_TIP][1]) / size
    up_b = (b[0][1] - b[INDEX_TIP][1]) / size

    # CROSSING: the left hand's finger base is left of the right hand's, but
    # if the fingers cross, the left hand's fingertips end up RIGHT of the
    # right hand's. Hands pressed flat together (praying pose) don't do this.
    base_a, base_b = a[[5, 9]].mean(axis=0), b[[5, 9]].mean(axis=0)
    tip_a, tip_b = a[[INDEX_TIP, MIDDLE_TIP]].mean(axis=0), b[[INDEX_TIP, MIDDLE_TIP]].mean(axis=0)
    crossed = (base_a[0] - base_b[0]) * (tip_a[0] - tip_b[0]) < 0

    # angle between the two fingers' directions (debug only for now)
    dir_a, dir_b = tip_a - base_a, tip_b - base_b
    cosang = np.dot(dir_a, dir_b) / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))

    # hand shape: index+middle extended, ring+pinky curled, on both hands
    fingers_l, fingers_r = finger_matches(a), finger_matches(b)

    checks = {
        "palms_close": palm_d < PALM_CLOSE,
        "tips_close": tip_d < TIPS_CLOSE,
        "a_up": up_a > UP_MARGIN,
        "b_up": up_b > UP_MARGIN,
    }
    score = sum(checks.values()) / len(checks)
    # mandatory gates: fingers must cross, tips must overlap, and the two
    # fingers must not point in opposite directions (rules out clasped hands)
    if not (crossed and checks["tips_close"] and angle < MAX_ANGLE
            and fingers_l + fingers_r >= FINGER_MIN):
        score = 0.0

    debug = {
        "hands": 2,
        "palm_d": round(palm_d, 2),
        "tip_d": round(tip_d, 2),
        "up_left": round(up_a, 2),
        "up_right": round(up_b, 2),
        "crossed": bool(crossed),
        "angle": int(angle),
        "fingers_L": f"{fingers_l}/4",
        "fingers_R": f"{fingers_r}/4",
    }
    return score, debug


def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    smooth_score = 0.0
    hold_start = None       # when the sign first appeared
    last_seen = 0.0         # last time the sign was present
    last_trigger = 0.0
    flash_until = 0.0

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            now = time.time()

            results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            hands_px = []
            if results.multi_hand_landmarks:
                for hl in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)
                    hands_px.append(to_px(hl, w, h))

            raw_score, debug = analyze(hands_px)
            smooth_score = SMOOTHING * raw_score + (1 - SMOOTHING) * smooth_score
            present = smooth_score >= SCORE_ON

            # hold timer with flicker grace period
            if present:
                last_seen = now
                if hold_start is None:
                    hold_start = now
            elif hold_start is not None and now - last_seen > GRACE_SECONDS:
                hold_start = None

            held = (now - hold_start) if hold_start else 0.0

            if held >= HOLD_SECONDS and now - last_trigger > COOLDOWN_SECONDS:
                print("SIGN DETECTED")
                last_trigger = now
                flash_until = now + 1.0
                hold_start = None

            # ---------- on-screen debug ----------
            bar_w = int(300 * min(held / HOLD_SECONDS, 1.0))
            cv2.rectangle(frame, (20, 20), (320, 45), (60, 60, 60), 2)
            cv2.rectangle(frame, (20, 20), (20 + bar_w, 45), (0, 200, 255), -1)
            cv2.putText(frame, f"score {smooth_score:.2f}", (340, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            y = 80
            for k, v in debug.items():
                cv2.putText(frame, f"{k}: {v}", (20, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y += 28

            if now < flash_until:
                cv2.putText(frame, "SIGN DETECTED!", (w // 2 - 250, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 140, 255), 5)

            cv2.imshow("Shadow Clone Jutsu - Phase 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
