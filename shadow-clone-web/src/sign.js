/**
 * Sign detection + the state machine around it.
 * A direct port of sign_detect.py and the logic in shadow_clone.py, so the
 * thresholds you tuned there behave the same way here.
 *
 * Everything here is pure JS with no browser APIs, so it can be unit-tested
 * in Node and reused outside the demo page.
 */

// ---------------- THRESHOLDS (same values as the Python version) ----------------
export const CFG = {
  // two-handed Shadow Clone sign
  PALM_CLOSE: 1.9,       // palm centers closer than this many "hand sizes"
  TIPS_CLOSE: 0.8,       // index/middle fingertips of the two hands this close
  UP_MARGIN: 0.05,       // fingertip must be this many hand sizes above the wrist
  MAX_ANGLE: 150,        // max angle (deg) between the two hands' finger directions
  EXT_RATIO: 1.5,        // finger is "extended" if tip is this much farther from wrist than its base
  FINGER_MIN: 6,         // out of 8 (4 fingers x 2 hands) must match the sign's finger shape
  HOLD_SECONDS: 0.5,     // how long the sign must be held to trigger
  GRACE_SECONDS: 0.3,    // flicker tolerance
  SMOOTHING: 0.4,        // 0..1, higher = reacts faster
  SCORE_ON: 0.6,         // smoothed score needed to count as "sign present"
  COOLDOWN_SECONDS: 2.0,
  SPAWN_SOUND_LEAD: 0.2, // start the spawn sound this long BEFORE the clones appear

  // one-handed dismiss sign (index + middle up)
  DISMISS_HOLD: 0.6,
  DISMISS_GRACE: 0.3,
  DISMISS_LOCKOUT: 1.5,  // ignore the dismiss pose this long after clones appear
  POINT_UP_MIN: 0.7,     // fingertips this many hand sizes above the wrist
  OTHER_HAND_FAR: 2.5,   // other hand must be at least this many hand sizes away
};

const INDEX_TIP = 8;
const MIDDLE_TIP = 12;
const PALM_POINTS = [0, 5, 9, 13, 17];
// [tip, base, shouldBeExtended]: index + middle out, ring + pinky curled
const FINGER_PATTERN = [[8, 5, true], [12, 9, true], [16, 13, false], [20, 17, false]];

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const mean2 = (pts) => [
  pts.reduce((s, p) => s + p[0], 0) / pts.length,
  pts.reduce((s, p) => s + p[1], 0) / pts.length,
];
const round2 = (x) => Math.round(x * 100) / 100;

/** MediaPipe normalized landmarks -> pixel coordinates [[x, y], ...]. */
export function toPx(landmarks, w, h) {
  return landmarks.map((l) => [l.x * w, l.y * h]);
}

export const handSize = (p) => dist(p[0], p[9]) + 1e-6; // wrist -> middle finger base
export const palmCenter = (p) => mean2(PALM_POINTS.map((i) => p[i]));

/** How many of the 4 fingers match the sign's hand shape (0..4). */
export function fingerMatches(p, cfg = CFG) {
  let n = 0;
  for (const [tip, base, wantExtended] of FINGER_PATTERN) {
    const ratio = dist(p[tip], p[0]) / (dist(p[base], p[0]) + 1e-6);
    if ((ratio > cfg.EXT_RATIO) === wantExtended) n++;
  }
  return n;
}

/** Score the current frame for the two-handed sign. Returns { score, debug }. */
export function analyze(handsPx, cfg = CFG) {
  if (handsPx.length < 2) return { score: 0, debug: { hands: handsPx.length } };

  // sort left-to-right so `a` is always the left hand (MediaPipe's order can swap)
  const [a, b] = handsPx.slice(0, 2).sort((p, q) => palmCenter(p)[0] - palmCenter(q)[0]);
  const size = (handSize(a) + handSize(b)) / 2;

  const palmD = dist(palmCenter(a), palmCenter(b)) / size;

  let tipD = Infinity;
  for (const i of [INDEX_TIP, MIDDLE_TIP]) {
    for (const j of [INDEX_TIP, MIDDLE_TIP]) tipD = Math.min(tipD, dist(a[i], b[j]));
  }
  tipD /= size;

  const upA = (a[0][1] - a[INDEX_TIP][1]) / size;
  const upB = (b[0][1] - b[INDEX_TIP][1]) / size;

  // CROSSING: the left hand's finger base is left of the right hand's, but if
  // the fingers cross, the left hand's fingertips end up RIGHT of the right's.
  const baseA = mean2([a[5], a[9]]);
  const baseB = mean2([b[5], b[9]]);
  const tipA = mean2([a[INDEX_TIP], a[MIDDLE_TIP]]);
  const tipB = mean2([b[INDEX_TIP], b[MIDDLE_TIP]]);
  const crossed = (baseA[0] - baseB[0]) * (tipA[0] - tipB[0]) < 0;

  const dirA = [tipA[0] - baseA[0], tipA[1] - baseA[1]];
  const dirB = [tipB[0] - baseB[0], tipB[1] - baseB[1]];
  const cosang =
    (dirA[0] * dirB[0] + dirA[1] * dirB[1]) /
    (Math.hypot(...dirA) * Math.hypot(...dirB) + 1e-6);
  const angle = (Math.acos(Math.max(-1, Math.min(1, cosang))) * 180) / Math.PI;

  const fingersL = fingerMatches(a, cfg);
  const fingersR = fingerMatches(b, cfg);

  const checks = {
    palmsClose: palmD < cfg.PALM_CLOSE,
    tipsClose: tipD < cfg.TIPS_CLOSE,
    aUp: upA > cfg.UP_MARGIN,
    bUp: upB > cfg.UP_MARGIN,
  };
  let score = Object.values(checks).filter(Boolean).length / 4;
  // mandatory gates
  if (!(crossed && checks.tipsClose && angle < cfg.MAX_ANGLE && fingersL + fingersR >= cfg.FINGER_MIN)) {
    score = 0;
  }

  return {
    score,
    debug: {
      hands: 2,
      palm_d: round2(palmD),
      tip_d: round2(tipD),
      up_left: round2(upA),
      up_right: round2(upB),
      crossed,
      angle: Math.trunc(angle),
      fingers_L: `${fingersL}/4`,
      fingers_R: `${fingersR}/4`,
    },
  };
}

/** One-hand sign: index + middle extended and pointing up, ring + pinky curled. */
export function twoFingerPose(p, cfg = CFG) {
  const wrist = p[0];
  const size = dist(p[0], p[9]) + 1e-6;
  const ratio = (t, b) => dist(p[t], wrist) / (dist(p[b], wrist) + 1e-6);

  const extended = [[8, 5], [12, 9]].every(([t, b]) => ratio(t, b) > cfg.EXT_RATIO);
  const curled = [[16, 13], [20, 17]].every(([t, b]) => ratio(t, b) < cfg.EXT_RATIO);
  const up = [8, 12].every((t) => (wrist[1] - p[t][1]) / size > cfg.POINT_UP_MIN);
  return extended && curled && up;
}

/** True if one hand makes the two-finger sign and the other hand (if any) is well away. */
export function dismissPosePresent(handsPx, cfg = CFG) {
  for (let i = 0; i < handsPx.length; i++) {
    const p = handsPx[i];
    if (!twoFingerPose(p, cfg)) continue;
    const size = dist(p[0], p[9]) + 1e-6;
    const othersFar = handsPx.every((q, j) => j === i || dist(p[0], q[0]) / size > cfg.OTHER_HAND_FAR);
    if (othersFar) return true;
  }
  return false;
}

/**
 * Turns per-frame hand positions into events: spawn sound / spawn / dismiss.
 * Also owns the clone on/off state so fade timing lives in one place.
 * All times are in seconds.
 */
export class SignController {
  constructor(overrides = {}) {
    this.cfg = { ...CFG, ...overrides };
    this.reset();
  }

  reset() {
    this.smoothScore = 0;
    this.holdStart = null;
    this.lastSeen = 0;
    this.lastTrigger = -Infinity;
    this.soundFired = false;
    this.poseStart = null;
    this.poseLastSeen = 0;
    this.clonesActive = false;
    this.spawnTime = 0;
    this.dismissTime = null;
  }

  /** Call once per frame. Returns what happened plus values for the debug overlay. */
  update(handsPx, now) {
    const c = this.cfg;
    const events = { spawnSound: false, spawn: false, dismissSound: false, dismiss: false };

    // ---- two-handed clone sign ----
    const { score: raw, debug } = analyze(handsPx, c);
    this.smoothScore = c.SMOOTHING * raw + (1 - c.SMOOTHING) * this.smoothScore;
    if (this.smoothScore >= c.SCORE_ON) {
      this.lastSeen = now;
      if (this.holdStart === null) this.holdStart = now;
    } else if (this.holdStart !== null && now - this.lastSeen > c.GRACE_SECONDS) {
      this.holdStart = null;
    }
    const held = this.holdStart !== null ? now - this.holdStart : 0;
    if (this.holdStart === null) this.soundFired = false;

    const ready = !this.clonesActive && now - this.lastTrigger > c.COOLDOWN_SECONDS;

    // start the sound slightly early so it lines up with the clones
    if (this.holdStart !== null && !this.soundFired && ready && held >= c.HOLD_SECONDS - c.SPAWN_SOUND_LEAD) {
      events.spawnSound = true;
      this.soundFired = true;
    }
    if (held >= c.HOLD_SECONDS && ready) {
      events.spawn = true;
      if (!this.soundFired) events.spawnSound = true;
      this.soundFired = false;
      this.clonesActive = true;
      this.spawnTime = now;
      this.dismissTime = null;
      this.lastTrigger = now;
      this.holdStart = null;
    }

    // ---- one-handed dismiss sign ----
    const poseHit = raw === 0 && dismissPosePresent(handsPx, c);
    let poseHeld = 0;
    if (this.clonesActive && this.dismissTime === null && now - this.spawnTime > c.DISMISS_LOCKOUT) {
      if (poseHit) {
        this.poseLastSeen = now;
        if (this.poseStart === null) this.poseStart = now;
      } else if (this.poseStart !== null && now - this.poseLastSeen > c.DISMISS_GRACE) {
        this.poseStart = null;
      }
      poseHeld = this.poseStart !== null ? now - this.poseStart : 0;
      if (poseHeld >= c.DISMISS_HOLD) {
        events.dismiss = true;
        events.dismissSound = true;
        this.dismissTime = now;
        this.poseStart = null;
      }
    } else {
      this.poseStart = null;
    }

    return { events, held, poseHit, poseHeld, smoothScore: this.smoothScore, debug };
  }

  /** Clone opacity 0..1 (fade in / fade out). Also switches clones off when the fade-out ends. */
  strength(now, fadeIn = 0.5, fadeOut = 0.4) {
    if (!this.clonesActive) return 0;
    if (this.dismissTime === null) return Math.min(1, (now - this.spawnTime) / fadeIn);
    const s = 1 - (now - this.dismissTime) / fadeOut;
    if (s <= 0) {
      this.clonesActive = false;
      this.dismissTime = null;
      return 0;
    }
    return s;
  }

  /** Dismiss now (button / key). Returns true if there was something to dismiss. */
  manualDismiss(now) {
    if (!this.clonesActive || this.dismissTime !== null) return false;
    this.dismissTime = now;
    this.poseStart = null;
    return true;
  }

  /** Spawn now without the hand sign (test button). Returns true if it spawned. */
  forceSpawn(now) {
    if (this.clonesActive) return false;
    this.clonesActive = true;
    this.spawnTime = now;
    this.dismissTime = null;
    this.lastTrigger = now;
    this.holdStart = null;
    this.soundFired = false;
    return true;
  }
}
