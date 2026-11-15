/**
 * ShadowCloneEngine: webcam -> hand-sign detection -> clones, smoke and sound.
 * Everything runs in the visitor's browser (MediaPipe Tasks Vision, WebGL/WASM);
 * nothing is uploaded and no backend is needed.
 *
 *   const engine = new ShadowCloneEngine({ canvas: document.querySelector('canvas') });
 *   button.onclick = () => engine.start();   // must be called from a user gesture
 *
 * See README.md for all options and how to embed it in an existing app.
 */
import { SignController, toPx } from './sign.js';
import { SoundBank } from './audio.js';
import { Smoke } from './smoke.js';
import { CloneCompositor, cloneBoxes } from './compositor.js';
import { maskToRGBA } from './mask.js';

const TASKS_VERSION = '0.10.14';
const DEFAULTS = {
  width: 960,                       // internal working size (same as the Python version)
  height: 540,
  threshold: 0.65,                  // person-mask threshold
  debug: true,                      // hand skeletons + progress bars
  fadeIn: 0.5,
  fadeOut: 0.4,
  trimSilence: false,               // skip leading silence in the sound files
  sounds: {
    spawn: 'assets/shadow_clone_sound.mp3',
    dismiss: 'assets/shadow_clone_poof.mp3',
  },
  // Self-host these for production (see README).
  visionUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/vision_bundle.mjs`,
  wasmUrl: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${TASKS_VERSION}/wasm`,
  handModelUrl:
    'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  segModelUrl:
    'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite',
  signConfig: {},                   // override any threshold from sign.js CFG
  onStatus: null,                   // (message: string) => void
  onEvent: null,                    // ({ type: 'spawn' | 'dismiss' }) => void
};

const makeCanvas = (w, h) => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

export class ShadowCloneEngine {
  /**
   * @param {object} options  `canvas` is required; everything else is optional (see DEFAULTS).
   */
  constructor(options) {
    if (!options || !options.canvas) throw new Error('ShadowCloneEngine needs a canvas');
    this.opts = { ...DEFAULTS, ...options, sounds: { ...DEFAULTS.sounds, ...(options.sounds || {}) } };
    const { width: W, height: H } = this.opts;
    this.W = W;
    this.H = H;

    this.canvas = options.canvas;
    this.canvas.width = W;
    this.canvas.height = H;
    this.ctx = this.canvas.getContext('2d');

    this.video = options.video || document.createElement('video');
    this.video.playsInline = true;
    this.video.muted = true;

    this.frameCanvas = makeCanvas(W, H);     // mirrored camera frame, what the models see
    this.frameCtx = this.frameCanvas.getContext('2d');
    this.alphaCanvas = makeCanvas(W, H);     // person mask at output size
    this.alphaCtx = this.alphaCanvas.getContext('2d');
    this.maskCanvas = null;                  // person mask at the model's native size
    this.maskImg = null;

    this.ctrl = new SignController(this.opts.signConfig);
    this.sounds = new SoundBank();
    this.smoke = new Smoke();
    this.comp = new CloneCompositor(W, H);

    this.threshold = this.opts.threshold;
    this.debug = this.opts.debug;
    this.invert = false;
    this.personBox = null;
    this.burstPending = false;
    this.handsPx = [];
    this.running = false;
    this.ready = false;

    this._badMask = 0;
    this._errors = 0;
    this._lastTs = 0;
    this._lastVideoT = -1;
    this._fps = 0;
    this._fpsFrames = 0;
    this._fpsT = 0;
    this._tick = this._tick.bind(this);
  }

  // ----------------------------- public API -----------------------------

  /** Load models + sounds, open the camera, start the loop. Call from a click handler. */
  async start() {
    if (this.running) return;
    this.sounds.init(); // synchronous part must happen inside the user gesture
    await Promise.all([this._init(), this._loadSounds()]);

    this._status('Starting camera…');
    this.stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    this.video.srcObject = this.stream;
    await this.video.play();

    this.ctrl.reset();
    this.running = true;
    this._errors = 0;
    this._status('Ready. Make the Shadow Clone sign!');
    this.raf = requestAnimationFrame(this._tick);
  }

  /** Stop the camera and the loop (models stay loaded so start() is fast next time). */
  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    this.stream = null;
    this.video.srcObject = null;
    this._status('Stopped.');
  }

  /** Free everything, including the models and the audio context. */
  destroy() {
    this.stop();
    this.hands?.close();
    this.segmenter?.close();
    this.hands = this.segmenter = null;
    this.ready = false;
    this.sounds.close();
  }

  /** Dismiss the clones (same as the two-finger sign). */
  dismiss() {
    const now = performance.now() / 1000;
    if (!this.ctrl.manualDismiss(now)) return;
    this.sounds.play('dismiss');
    this.burstPending = true;
    this._emit('dismiss');
  }

  /** Spawn the clones without the hand sign (handy for testing). */
  spawn() {
    const now = performance.now() / 1000;
    if (!this.ctrl.forceSpawn(now)) return;
    this.sounds.play('spawn');
    this.burstPending = true;
    this._emit('spawn');
  }

  setThreshold(v) { this.threshold = Math.min(0.95, Math.max(0.1, v)); }
  setSpawnSoundLead(seconds) { this.ctrl.cfg.SPAWN_SOUND_LEAD = Math.max(0, seconds); }
  toggleDebug() { this.debug = !this.debug; return this.debug; }
  toggleInvert() { this.invert = !this.invert; return this.invert; }
  get fps() { return this._fps; }
  get clonesActive() { return this.ctrl.clonesActive; }

  // ------------------------------ loading ------------------------------

  async _init() {
    if (this.ready) return;
    this._status('Loading hand and body models…');
    const vision = await import(/* @vite-ignore */ this.opts.visionUrl);
    this.vision = vision;
    const fileset = await vision.FilesetResolver.forVisionTasks(this.opts.wasmUrl);

    this.hands = await this._create(vision.HandLandmarker, fileset, this.opts.handModelUrl, {
      runningMode: 'VIDEO',
      numHands: 2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });
    this.segmenter = await this._create(vision.ImageSegmenter, fileset, this.opts.segModelUrl, {
      runningMode: 'VIDEO',
      outputConfidenceMasks: true,
      outputCategoryMask: false,
    });
    this.ready = true;
  }

  /** Try the GPU first, fall back to CPU (some browsers/drivers can't do GPU). */
  async _create(Task, fileset, modelAssetPath, extra) {
    let lastErr;
    for (const delegate of ['GPU', 'CPU']) {
      try {
        return await Task.createFromOptions(fileset, { baseOptions: { modelAssetPath, delegate }, ...extra });
      } catch (err) {
        lastErr = err;
        console.warn(`MediaPipe task failed on ${delegate}:`, err);
      }
    }
    throw lastErr;
  }

  async _loadSounds() {
    const { sounds, trimSilence } = this.opts;
    const results = await Promise.all([
      this.sounds.load('spawn', sounds.spawn, { trimSilence }),
      this.sounds.load('dismiss', sounds.dismiss, { trimSilence }),
    ]);
    if (results.includes(false)) {
      this._status('Sound files not found, running silent. (Put the mp3s in assets/.)');
    }
  }

  // ------------------------------ main loop ------------------------------

  _tick() {
    if (!this.running) return;
    this.raf = requestAnimationFrame(this._tick);
    const v = this.video;
    if (v.readyState < 2 || v.currentTime === this._lastVideoT) return; // no new camera frame yet
    this._lastVideoT = v.currentTime;
    try {
      this._processFrame();
    } catch (err) {
      console.error(err);
      if (++this._errors >= 5) {
        this._status(`Stopped after repeated errors: ${err.message || err}`);
        this.stop();
      }
    }
  }

  _processFrame() {
    const { W, H } = this;
    const perf = performance.now();
    const now = perf / 1000;

    // mirrored frame, so it feels like a mirror and matches the Python version's geometry
    const fc = this.frameCtx;
    fc.save();
    fc.translate(W, 0);
    fc.scale(-1, 1);
    fc.drawImage(this.video, 0, 0, W, H);
    fc.restore();

    // MediaPipe wants strictly increasing timestamps
    const ts = Math.max(perf, this._lastTs + 1);
    this._lastTs = ts;

    // ---- hands -> sign logic ----
    const hr = this.hands.detectForVideo(this.frameCanvas, ts);
    this.handsPx = (hr.landmarks || []).map((l) => toPx(l, W, H));
    const st = this.ctrl.update(this.handsPx, now);
    if (st.events.spawnSound) this.sounds.play('spawn');
    if (st.events.spawn) { this.burstPending = true; this._emit('spawn'); }
    if (st.events.dismissSound) this.sounds.play('dismiss');
    if (st.events.dismiss) { this.burstPending = true; this._emit('dismiss'); }
    const strength = this.ctrl.strength(now, this.opts.fadeIn, this.opts.fadeOut);

    // ---- person mask ----
    this.segmenter.segmentForVideo(this.frameCanvas, ts, (result) => this._ingestMask(result));

    // ---- draw ----
    this.comp.render(this.ctx, this.frameCanvas, this.alphaCanvas, strength);
    if (this.burstPending) {
      this.smoke.burst(cloneBoxes(this.personBox, W, H), now);
      this.burstPending = false;
    }
    this.smoke.draw(this.ctx, this.comp.cutout, now);
    if (this.debug) this._drawOverlay(st);

    if (!this._fpsT) this._fpsT = perf;
    this._fpsFrames++;
    if (perf - this._fpsT >= 1000) {
      this._fps = Math.round((this._fpsFrames * 1000) / (perf - this._fpsT));
      this._fpsFrames = 0;
      this._fpsT = perf;
    }
  }

  _maskIndex(count) {
    if (count === 1) return 0;
    if (this._maskIdx !== undefined && this._maskIdx < count) return this._maskIdx;
    const labels = (this.segmenter.getLabels && this.segmenter.getLabels()) || [];
    let i = labels.findIndex((l) => /person|selfie|foreground|body/i.test(l));
    if (i < 0) {
      const bg = labels.findIndex((l) => /background/i.test(l));
      i = bg === 0 ? 1 : bg > 0 ? 0 : 1;
    }
    this._maskIdx = Math.min(i, count - 1);
    return this._maskIdx;
  }

  _ingestMask(result) {
    const masks = result.confidenceMasks;
    if (!masks || !masks.length) return;
    const m = masks[this._maskIndex(masks.length)];
    const mw = m.width, mh = m.height;
    const data = m.getAsFloat32Array();

    if (!this.maskCanvas || this.maskCanvas.width !== mw || this.maskCanvas.height !== mh) {
      this.maskCanvas = makeCanvas(mw, mh);
      this.maskCtx = this.maskCanvas.getContext('2d');
      this.maskImg = this.maskCtx.createImageData(mw, mh);
    }
    const { box, stats } = maskToRGBA(data, mw, mh, this.threshold, this.maskImg.data, this.invert);
    this.maskCtx.putImageData(this.maskImg, 0, 0);

    const a = this.alphaCtx;
    a.clearRect(0, 0, this.W, this.H);
    a.imageSmoothingEnabled = true;
    a.drawImage(this.maskCanvas, 0, 0, this.W, this.H);

    this.personBox = box && {
      x0: (box.x0 / mw) * this.W, x1: (box.x1 / mw) * this.W,
      y0: (box.y0 / mh) * this.H, y1: (box.y1 / mh) * this.H,
    };

    // Safety net: if the "person" is the background (corners solid, centre empty) for
    // ~20 frames in a row, the model's mask is the wrong way round, so flip it.
    this._badMask = stats.corner > 0.7 && stats.center < 0.3 ? this._badMask + 1 : 0;
    if (this._badMask >= 20) {
      this.invert = !this.invert;
      this._badMask = 0;
      console.info('Person mask looked inverted; flipped it.');
    }
  }

  // ------------------------------ overlay ------------------------------

  _drawOverlay(st) {
    const ctx = this.ctx;
    const cfg = this.ctrl.cfg;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;

    // hand skeletons
    const conns = this.vision?.HandLandmarker?.HAND_CONNECTIONS || [];
    ctx.lineWidth = 2;
    ctx.strokeStyle = 'rgba(255,255,255,0.85)';
    ctx.fillStyle = '#ff3b3b';
    for (const p of this.handsPx) {
      ctx.beginPath();
      for (const { start, end } of conns) {
        ctx.moveTo(p[start][0], p[start][1]);
        ctx.lineTo(p[end][0], p[end][1]);
      }
      ctx.stroke();
      for (const [x, y] of p) {
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const bar = (y, frac, color) => {
      ctx.strokeStyle = '#3c3c3c';
      ctx.lineWidth = 2;
      ctx.strokeRect(20, y, 300, 25);
      ctx.fillStyle = color;
      ctx.fillRect(20, y, 300 * Math.min(1, frac), 25);
    };
    ctx.font = '600 16px system-ui, sans-serif';
    ctx.textBaseline = 'middle';

    bar(20, st.held / cfg.HOLD_SECONDS, '#ffc800');
    ctx.fillStyle = '#fff';
    ctx.fillText(`score ${st.smoothScore.toFixed(2)}  thr ${this.threshold.toFixed(2)}  ${this._fps} fps`, 335, 33);

    let label = 'make the sign';
    if (this.ctrl.clonesActive) {
      bar(55, st.poseHeld / cfg.DISMISS_HOLD, '#ff3c3c');
      ctx.fillStyle = '#fff';
      ctx.fillText(`two fingers up: ${st.poseHit ? 'True' : 'False'}`, 335, 68);
      label = 'CLONES ON - raise index + middle finger to dismiss';
    }
    ctx.fillStyle = '#3ddc84';
    ctx.fillText(label, 20, 100);
    ctx.restore();
  }

  // ------------------------------ helpers ------------------------------

  _status(msg) { this.opts.onStatus?.(msg); }
  _emit(type) { this.opts.onEvent?.({ type }); }
}
