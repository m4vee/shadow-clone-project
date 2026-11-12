"""
Shadow Clone Jutsu project - Phase 5
- Make the Shadow Clone sign -> copies of you appear (plays shadow_clone_sound.mp3)
- Raise your INDEX + MIDDLE fingers on one hand (ring + pinky curled) -> clones
  vanish (plays shadow_clone_poof.mp3)

Puffs of smoke burst over the clones when they appear and when they vanish.

Needs sign_detect.py, shadow_clone_sound.mp3 and shadow_clone_poof.mp3 in the
same folder as this script. MP3 playback uses Windows' built-in player (no install).

Keys:  q = quit
       c = dismiss the clones (backup for the finger sign)
       d = toggle debug overlay (hand skeletons + detector numbers)
       [ / ] = lower / raise the segmentation threshold
"""

import ctypes
import os
import time

import cv2
import numpy as np
import mediapipe as mp

from sign_detect import (
    analyze, to_px,
    HOLD_SECONDS, GRACE_SECONDS, SMOOTHING, SCORE_ON, COOLDOWN_SECONDS,
    EXT_RATIO,
)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_seg = mp.solutions.selfie_segmentation

WORK_W, WORK_H = 960, 540      # everything runs at this size for speed
FADE_IN = 0.5                  # seconds for clones to fade in
FADE_OUT = 0.4                 # seconds for clones to fade out

# ------------- dismiss sign (index + middle fingers up) -------------
DISMISS_HOLD = 0.6       # hold the pose this long to dismiss
DISMISS_GRACE = 0.3      # flicker tolerance, same idea as the clone sign
DISMISS_LOCKOUT = 1.5    # ignore the dismiss pose this long after clones appear
POINT_UP_MIN = 0.7       # fingertips must be this many hand sizes above the wrist
OTHER_HAND_FAR = 2.5     # the other hand must be at least this many hand sizes away
# ---------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
SPAWN_SOUND = os.path.join(HERE, "shadow_clone_sound.mp3")
DISMISS_SOUND = os.path.join(HERE, "shadow_clone_poof.mp3")

# Play the spawn sound this many seconds BEFORE the clones appear, to make up
# for any silence at the start of the mp3. Raise it if the sound still feels
# late, lower it (0 = exactly with the clones) if it now feels early.
SPAWN_SOUND_LEAD = 0.2

# ------------------------------ smoke ------------------------------
SMOKE_PUFFS = 50         # 14 puffs of smoke per clone
SMOKE_LIFE = (0.7, 1.1)  # seconds each puff lasts (min, max)
SMOKE_OPACITY = 0.9      # peak opacity of a puff (0..1)
SMOKE_COLOR = (235, 235, 235)   # BGR, light grey

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


# ------------------------------ sound ------------------------------
# Windows' built-in MCI player handles mp3 with no extra packages.
try:
    _mci = ctypes.windll.winmm.mciSendStringW
except AttributeError:      # not Windows
    _mci = None


_loaded = set()


def load_sound(path, alias):
    """Open an mp3 once at startup. Opening it at play time is what causes lag."""
    if _mci is None or not os.path.exists(path):
        return
    if _mci(f'open "{path}" type mpegvideo alias {alias}', None, 0, 0) == 0:
        _loaded.add(alias)
    else:
        print(f"Note: could not open {os.path.basename(path)} for playback")


def play_sound(alias):
    """Play a preloaded sound from the start without blocking the video loop."""
    if alias in _loaded:
        _mci(f"play {alias} from 0", None, 0, 0)


def play_spawn():
    play_sound("clone_spawn")


def play_dismiss():
    play_sound("clone_dismiss")


def close_sounds():
    for alias in list(_loaded):
        _mci(f"close {alias}", None, 0, 0)
    _loaded.clear()


# ------------------------------ smoke ------------------------------
def make_sprites(n=4, size=128):
    """A few soft, cloudy puff shapes (float 0..1)."""
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[-1:1:size * 1j, -1:1:size * 1j]
    radial = np.clip(1 - np.sqrt(xx ** 2 + yy ** 2), 0, 1) ** 0.8
    sprites = []
    for _ in range(n):
        noise = cv2.GaussianBlur(rng.random((size, size)).astype(np.float32), (0, 0), size / 10)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        sprites.append(np.clip(radial * (0.55 + 0.6 * noise), 0, 1).astype(np.float32))
    return sprites


class Smoke:
    """Puffs of smoke that burst over each clone, drift up, grow and fade."""

    def __init__(self):
        self.sprites = make_sprites()
        self.rng = np.random.default_rng()
        self.puffs = []

    def burst(self, boxes, now):
        rng = self.rng
        for x0, y0, x1, y1 in boxes:
            bw, bh = x1 - x0, y1 - y0
            for _ in range(SMOKE_PUFFS):
                r_end = rng.uniform(0.28, 0.45) * bw
                self.puffs.append({
                    "x": rng.uniform(x0 + 0.1 * bw, x1 - 0.1 * bw),
                    "y": rng.uniform(y0 + 0.05 * bh, y1 - 0.05 * bh),
                    "vx": rng.normal(0, 15),
                    "vy": rng.uniform(-50, -15),
                    "r0": 0.4 * r_end,
                    "r1": r_end,
                    "born": now,
                    "life": rng.uniform(*SMOKE_LIFE),
                    "sprite": self.sprites[rng.integers(len(self.sprites))],
                    "flip": bool(rng.integers(2)),
                })

    def apply(self, out, person_alpha, now):
        """Blend all live puffs over `out`, keeping the real you smoke-free."""
        self.puffs = [p for p in self.puffs if now - p["born"] < p["life"]]
        if not self.puffs:
            return out
        h, w = out.shape[:2]
        sa = np.zeros((h, w), np.float32)
        for p in self.puffs:
            age = now - p["born"]
            t = age / p["life"]
            radius = p["r0"] + (p["r1"] - p["r0"]) * (1 - (1 - t) ** 2)
            opacity = SMOKE_OPACITY * (1 - t) ** 1.5 * min(1.0, age / 0.08)
            d = int(2 * radius)
            if d < 4 or opacity <= 0.01:
                continue
            sp = cv2.resize(p["sprite"], (d, d))
            if p["flip"]:
                sp = sp[:, ::-1]
            cx = p["x"] + p["vx"] * age
            cy = p["y"] + p["vy"] * age
            x0, y0 = int(cx - d / 2), int(cy - d / 2)
            cx0, cy0 = max(x0, 0), max(y0, 0)
            cx1, cy1 = min(x0 + d, w), min(y0 + d, h)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            sub = sp[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] * opacity
            roi = sa[cy0:cy1, cx0:cx1]
            roi[:] = 1 - (1 - roi) * (1 - sub)
        sa = (sa * (1 - person_alpha))[..., None]
        color = np.array(SMOKE_COLOR, np.float32)
        return np.clip(out.astype(np.float32) * (1 - sa) + color * sa, 0, 255).astype(np.uint8)


def clone_boxes(alpha, w, h):
    """Where each clone ends up on screen, from the bounding box of your cutout."""
    cols = np.where(alpha.max(axis=0) > 0.5)[0]
    rows = np.where(alpha.max(axis=1) > 0.5)[0]
    if cols.size == 0 or rows.size == 0:
        return []
    bx0, bx1, by0, by1 = cols[0], cols[-1], rows[0], rows[-1]
    boxes = []
    for dx, s, _ in CLONES:
        ox, oy = (1 - s) * w / 2 + dx * w, (1 - s) * h
        boxes.append((s * bx0 + ox, s * by0 + oy, s * bx1 + ox, s * by1 + oy))
    return boxes


# ------------------------- dismiss pose -------------------------
def two_finger_pose(pts):
    """One-hand sign: index + middle extended and pointing up,
    ring + pinky curled."""
    wrist = pts[0]
    size = np.linalg.norm(wrist - pts[9]) + 1e-6

    def ratio(tip, base):
        return np.linalg.norm(pts[tip] - wrist) / (np.linalg.norm(pts[base] - wrist) + 1e-6)

    extended = all(ratio(t, b) > EXT_RATIO for t, b in [(8, 5), (12, 9)])
    curled = all(ratio(t, b) < EXT_RATIO for t, b in [(16, 13), (20, 17)])
    up = all((wrist[1] - pts[t][1]) / size > POINT_UP_MIN for t in (8, 12))
    return extended and curled and up


def dismiss_pose_present(hands_px):
    """True if one hand makes the two-finger sign and the other hand (if any)
    is well away from it, so it can't be confused with the two-handed clone sign."""
    for i, p in enumerate(hands_px):
        if not two_finger_pose(p):
            continue
        size = np.linalg.norm(p[0] - p[9]) + 1e-6
        others_far = all(
            np.linalg.norm(p[0] - q[0]) / size > OTHER_HAND_FAR
            for j, q in enumerate(hands_px) if j != i
        )
        if others_far:
            return True
    return False


# ----------------------- mask + compositing -----------------------
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
    for p in (SPAWN_SOUND, DISMISS_SOUND):
        if not os.path.exists(p):
            print(f"Note: sound file not found, skipping: {os.path.basename(p)}")
    if _mci is None:
        print("Note: sound needs Windows. Running without sound.")
    load_sound(SPAWN_SOUND, "clone_spawn")
    load_sound(DISMISS_SOUND, "clone_dismiss")
    smoke = Smoke()
    sound_fired = False
    burst_pending = False

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    threshold = 0.65
    debug_on = True

    # clone sign detection state
    smooth_score = 0.0
    hold_start = None
    last_seen = 0.0
    last_trigger = 0.0

    # dismiss sign state
    pose_start = None
    pose_last_seen = 0.0
    pose_hit = False

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

            # ---------- hand tracking ----------
            hres = hands.process(rgb)
            hands_px, hand_lms = [], []
            if hres.multi_hand_landmarks:
                for hl in hres.multi_hand_landmarks:
                    hand_lms.append(hl)
                    hands_px.append(to_px(hl, WORK_W, WORK_H))

            # ---------- clone sign (two hands) ----------
            raw_score, debug = analyze(hands_px)
            smooth_score = SMOOTHING * raw_score + (1 - SMOOTHING) * smooth_score
            if smooth_score >= SCORE_ON:
                last_seen = now
                if hold_start is None:
                    hold_start = now
            elif hold_start is not None and now - last_seen > GRACE_SECONDS:
                hold_start = None
            held = (now - hold_start) if hold_start else 0.0
            if hold_start is None:
                sound_fired = False

            # start the sound slightly early so it lines up with the clones
            if (hold_start and not sound_fired and not clones_active
                    and held >= HOLD_SECONDS - SPAWN_SOUND_LEAD
                    and now - last_trigger > COOLDOWN_SECONDS):
                play_spawn()
                sound_fired = True

            if (held >= HOLD_SECONDS and not clones_active
                    and now - last_trigger > COOLDOWN_SECONDS):
                print("SHADOW CLONE JUTSU!")
                clones_active = True
                spawn_time = now
                dismiss_time = None
                last_trigger = now
                hold_start = None
                if not sound_fired:
                    play_spawn()
                sound_fired = False
                burst_pending = True

            # ---------- dismiss sign (index + middle fingers up) ----------
            pose_hit = raw_score == 0 and dismiss_pose_present(hands_px)
            pose_held = 0.0
            if clones_active and dismiss_time is None and now - spawn_time > DISMISS_LOCKOUT:
                if pose_hit:
                    pose_last_seen = now
                    if pose_start is None:
                        pose_start = now
                elif pose_start is not None and now - pose_last_seen > DISMISS_GRACE:
                    pose_start = None
                pose_held = (now - pose_start) if pose_start else 0.0
                if pose_held >= DISMISS_HOLD:
                    print("Clones dismissed")
                    dismiss_time = now
                    pose_start = None
                    play_dismiss()
                    burst_pending = True
            else:
                pose_start = None

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

            # ---------- smoke ----------
            if burst_pending:
                smoke.burst(clone_boxes(alpha, WORK_W, WORK_H), now)
                burst_pending = False
            out = smoke.apply(out, alpha, now)

            # ---------- overlay ----------
            if debug_on:
                for hl in hand_lms:
                    mp_draw.draw_landmarks(out, hl, mp_hands.HAND_CONNECTIONS)
                bar = int(300 * min(held / HOLD_SECONDS, 1.0))
                cv2.rectangle(out, (20, 20), (320, 45), (60, 60, 60), 2)
                cv2.rectangle(out, (20, 20), (20 + bar, 45), (0, 200, 255), -1)
                cv2.putText(out, f"score {smooth_score:.2f}  thr {threshold:.2f}",
                            (335, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if clones_active:
                    bar2 = int(300 * min(pose_held / DISMISS_HOLD, 1.0))
                    cv2.rectangle(out, (20, 55), (320, 80), (60, 60, 60), 2)
                    cv2.rectangle(out, (20, 55), (20 + bar2, 80), (60, 60, 255), -1)
                    cv2.putText(out, f"two fingers up: {pose_hit}", (335, 77),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    state = "CLONES ON - raise index + middle finger to dismiss"
                else:
                    state = "make the sign"
                cv2.putText(out, state, (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Shadow Clone Jutsu - Phase 5", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") and clones_active and dismiss_time is None:
                dismiss_time = now
                play_dismiss()
                burst_pending = True
            elif key == ord("d"):
                debug_on = not debug_on
            elif key == ord("["):
                threshold = max(0.1, threshold - 0.05)
            elif key == ord("]"):
                threshold = min(0.95, threshold + 0.05)

    cap.release()
    cv2.destroyAllWindows()
    close_sounds()


if __name__ == "__main__":
    main()
