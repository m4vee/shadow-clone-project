/**
 * Puffs of smoke that burst over each clone, drift up, grow and fade.
 * Port of the Python Smoke class. Smoke is drawn on its own layer and the
 * real you is erased from that layer, so you never get smoked over.
 */

export const SMOKE = {
  PUFFS: 50,            // puffs per clone
  LIFE: [0.7, 1.1],     // seconds each puff lasts (min, max)
  OPACITY: 0.9,         // peak opacity of a puff (0..1)
  COLOR: [235, 235, 235],
};

const rand = (a, b) => a + Math.random() * (b - a);
function gauss() {
  let u = 0;
  while (!u) u = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random());
}

/** One pass of a separable box blur with clamped edges. */
function boxBlur(src, w, h, r) {
  const d = 2 * r + 1;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += src[y * w + Math.min(w - 1, Math.max(0, k))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / d;
      acc += src[y * w + Math.min(w - 1, x + r + 1)] - src[y * w + Math.max(0, x - r)];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / d;
      acc += tmp[Math.min(h - 1, y + r + 1) * w + x] - tmp[Math.max(0, y - r) * w + x];
    }
  }
  return out;
}

/** A few soft, cloudy puff shapes as small canvases (grey with varying alpha). */
function makeSprites(n = 4, size = 128) {
  const sprites = [];
  for (let s = 0; s < n; s++) {
    let noise = new Float32Array(size * size).map(() => Math.random());
    for (let pass = 0; pass < 3; pass++) noise = boxBlur(noise, size, size, Math.round(size / 16));
    let lo = Infinity, hi = -Infinity;
    for (const v of noise) { if (v < lo) lo = v; if (v > hi) hi = v; }

    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(size, size);
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        const nx = (x / (size - 1)) * 2 - 1, ny = (y / (size - 1)) * 2 - 1;
        const radial = Math.pow(Math.max(0, 1 - Math.hypot(nx, ny)), 0.8);
        const nv = (noise[y * size + x] - lo) / (hi - lo + 1e-6);
        const a = Math.min(1, Math.max(0, radial * (0.55 + 0.6 * nv)));
        const o = (y * size + x) * 4;
        img.data[o] = SMOKE.COLOR[0]; img.data[o + 1] = SMOKE.COLOR[1]; img.data[o + 2] = SMOKE.COLOR[2];
        img.data[o + 3] = a * 255;
      }
    }
    ctx.putImageData(img, 0, 0);
    sprites.push(canvas);
  }
  return sprites;
}

export class Smoke {
  constructor() {
    this.sprites = makeSprites();
    this.puffs = [];
    this.layer = null;
  }

  /** Spawn puffs over each box [x0, y0, x1, y1]. `now` is in seconds. */
  burst(boxes, now) {
    for (const [x0, y0, x1, y1] of boxes) {
      const bw = x1 - x0, bh = y1 - y0;
      for (let i = 0; i < SMOKE.PUFFS; i++) {
        const rEnd = rand(0.28, 0.45) * bw;
        this.puffs.push({
          x: rand(x0 + 0.1 * bw, x1 - 0.1 * bw),
          y: rand(y0 + 0.05 * bh, y1 - 0.05 * bh),
          vx: gauss() * 15,
          vy: rand(-50, -15),
          r0: 0.4 * rEnd,
          r1: rEnd,
          born: now,
          life: rand(...SMOKE.LIFE),
          sprite: this.sprites[Math.floor(Math.random() * this.sprites.length)],
          flip: Math.random() < 0.5,
        });
      }
    }
  }

  /** Draw all live puffs onto ctx, with the person (cutout canvas) erased from the smoke. */
  draw(ctx, cutoutCanvas, now) {
    this.puffs = this.puffs.filter((p) => now - p.born < p.life);
    if (!this.puffs.length) return;

    const { width: w, height: h } = ctx.canvas;
    if (!this.layer || this.layer.width !== w || this.layer.height !== h) {
      this.layer = document.createElement('canvas');
      this.layer.width = w;
      this.layer.height = h;
    }
    const lc = this.layer.getContext('2d');
    lc.globalCompositeOperation = 'source-over';
    lc.globalAlpha = 1;
    lc.clearRect(0, 0, w, h);

    for (const p of this.puffs) {
      const age = now - p.born;
      const t = age / p.life;
      const radius = p.r0 + (p.r1 - p.r0) * (1 - (1 - t) ** 2);
      const opacity = SMOKE.OPACITY * (1 - t) ** 1.5 * Math.min(1, age / 0.08);
      const d = 2 * radius;
      if (d < 4 || opacity <= 0.01) continue;
      lc.save();
      lc.globalAlpha = opacity;
      lc.translate(p.x + p.vx * age, p.y + p.vy * age);
      if (p.flip) lc.scale(-1, 1);
      lc.drawImage(p.sprite, -d / 2, -d / 2, d, d);
      lc.restore();
    }

    // keep the real you smoke-free
    lc.globalCompositeOperation = 'destination-out';
    lc.globalAlpha = 1;
    lc.drawImage(cutoutCanvas, 0, 0);
    lc.globalCompositeOperation = 'source-over';

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.drawImage(this.layer, 0, 0);
    ctx.restore();
  }
}
