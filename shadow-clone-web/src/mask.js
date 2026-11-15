/**
 * Turns MediaPipe's person-confidence mask into a clean, soft-edged alpha.
 * Port of clean_mask() from the Python version: threshold, drop floating
 * specks (keep only the big blob(s) that are you), soften the edge.
 *
 * Pure function over typed arrays, so it runs in Node for tests too.
 *
 * @param {Float32Array} conf       person confidence per pixel (0..1), w*h long
 * @param {number} w
 * @param {number} h
 * @param {number} threshold        value above which a pixel counts as "person"
 * @param {Uint8ClampedArray} out   RGBA buffer, w*h*4 (written: white + alpha)
 * @param {boolean} invert          treat the mask as background-confidence instead
 * @returns {{box: {x0,y0,x1,y1}|null, stats: {corner:number, center:number}}}
 */
export function maskToRGBA(conf, w, h, threshold, out, invert = false) {
  // ---- coarse grid (<= ~160 cells wide) used for speck removal ----
  const stride = Math.max(1, Math.floor(Math.max(w, h) / 160));
  const gw = Math.ceil(w / stride);
  const gh = Math.ceil(h / stride);
  const val = invert ? (i) => 1 - conf[i] : (i) => conf[i];

  const bin = new Uint8Array(gw * gh);
  let cornerSum = 0, cornerN = 0, centerSum = 0, centerN = 0;
  for (let gy = 0; gy < gh; gy++) {
    const y = Math.min(h - 1, gy * stride);
    for (let gx = 0; gx < gw; gx++) {
      const x = Math.min(w - 1, gx * stride);
      const v = val(y * w + x);
      bin[gy * gw + gx] = v > threshold ? 1 : 0;
      const fx = gx / gw, fy = gy / gh;
      if (fy < 0.3 && (fx < 0.12 || fx > 0.88)) { cornerSum += v; cornerN++; }
      else if (fx > 0.35 && fx < 0.65 && fy > 0.5 && fy < 0.95) { centerSum += v; centerN++; }
    }
  }

  // ---- connected components (8-connectivity), keep blobs >= 15% of the largest ----
  const labels = new Int32Array(gw * gh);
  const stack = new Int32Array(gw * gh);
  const areas = [0];
  let n = 0;
  for (let s = 0; s < gw * gh; s++) {
    if (!bin[s] || labels[s]) continue;
    n++;
    let sp = 0, area = 0;
    stack[sp++] = s;
    labels[s] = n;
    while (sp) {
      const c = stack[--sp];
      area++;
      const cx = c % gw, cy = (c / gw) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const ny = cy + dy;
        if (ny < 0 || ny >= gh) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = cx + dx;
          if ((!dx && !dy) || nx < 0 || nx >= gw) continue;
          const ni = ny * gw + nx;
          if (bin[ni] && !labels[ni]) { labels[ni] = n; stack[sp++] = ni; }
        }
      }
    }
    areas.push(area);
  }
  const maxArea = Math.max(0, ...areas);
  const keep = new Uint8Array(gw * gh);
  for (let i = 0; i < keep.length; i++) {
    const l = labels[i];
    keep[i] = l && areas[l] >= 0.15 * maxArea ? 1 : 0;
  }
  // grow the kept region by one cell so we never clip the true edge
  const keepD = new Uint8Array(gw * gh);
  for (let gy = 0; gy < gh; gy++) {
    for (let gx = 0; gx < gw; gx++) {
      let k = 0;
      for (let dy = -1; dy <= 1 && !k; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = gy + dy, nx = gx + dx;
          if (ny >= 0 && ny < gh && nx >= 0 && nx < gw && keep[ny * gw + nx]) { k = 1; break; }
        }
      }
      keepD[gy * gw + gx] = k;
    }
  }

  // ---- full-resolution soft alpha; a slightly high ramp trims the fringe (like erosion) ----
  const hi = Math.min(threshold + 0.12, 0.999);
  const lo = Math.min(threshold, hi - 0.05);
  const inv = 1 / (hi - lo);
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let y = 0; y < h; y++) {
    const row = ((y / stride) | 0) * gw;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = val(i);
      let a = 0;
      if (v > lo && keepD[row + ((x / stride) | 0)]) {
        let t = (v - lo) * inv;
        if (t > 1) t = 1;
        a = t * t * (3 - 2 * t);
      }
      const o = i * 4;
      out[o] = 255; out[o + 1] = 255; out[o + 2] = 255; out[o + 3] = a * 255;
      if (a > 0.5) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }

  return {
    box: x0 <= x1 ? { x0, y0, x1, y1, w, h } : null,
    stats: { corner: cornerN ? cornerSum / cornerN : 0, center: centerN ? centerSum / centerN : 0 },
  };
}
