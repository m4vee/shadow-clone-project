/**
 * Draws the clones behind you and the real you on top.
 * Same layout as the Python version: each clone is a scaled, dimmed copy of
 * your cutout, shifted sideways and anchored to the bottom-centre of the frame.
 */

// [x offset as fraction of frame width, scale, brightness], back-to-front
export const CLONES = [
  [-0.42, 0.80, 0.70],
  [0.42, 0.80, 0.70],
  [-0.30, 0.88, 0.80],
  [0.30, 0.88, 0.80],
  [-0.18, 0.94, 0.90],
  [0.18, 0.94, 0.90],
];

const makeCanvas = (w, h) => {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  return c;
};

export class CloneCompositor {
  constructor(w, h, clones = CLONES) {
    this.w = w;
    this.h = h;
    this.clones = clones;
    this.cutout = makeCanvas(w, h); // you, with transparent background
    this.cutoutCtx = this.cutout.getContext('2d');
    // one dimmed copy per distinct brightness (cheaper than one per clone)
    this.dim = new Map();
    for (const b of new Set(clones.map((c) => c[2]))) {
      const canvas = makeCanvas(w, h);
      this.dim.set(b, { canvas, ctx: canvas.getContext('2d') });
    }
  }

  /**
   * @param {CanvasRenderingContext2D} ctx   output
   * @param {CanvasImageSource} frame        mirrored camera frame (w x h)
   * @param {CanvasImageSource} alpha        person mask: opaque where you are (w x h)
   * @param {number} strength                0..1 clone opacity
   */
  render(ctx, frame, alpha, strength) {
    const { w, h, cutoutCtx: cc } = this;

    // you, cut out of the frame
    cc.globalCompositeOperation = 'source-over';
    cc.clearRect(0, 0, w, h);
    cc.drawImage(frame, 0, 0, w, h);
    cc.globalCompositeOperation = 'destination-in';
    cc.drawImage(alpha, 0, 0, w, h);
    cc.globalCompositeOperation = 'source-over';

    // background = the live frame
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = 'source-over';
    ctx.drawImage(frame, 0, 0, w, h);

    if (strength > 0) {
      for (const [b, { canvas, ctx: dc }] of this.dim) {
        dc.globalCompositeOperation = 'source-over';
        dc.clearRect(0, 0, w, h);
        dc.drawImage(this.cutout, 0, 0);
        dc.globalCompositeOperation = 'source-atop';
        dc.fillStyle = `rgba(0,0,0,${1 - b})`;
        dc.fillRect(0, 0, w, h);
        dc.globalCompositeOperation = 'source-over';
      }
      for (const [dx, s, b] of this.clones) {
        ctx.save();
        ctx.globalAlpha = Math.min(1, strength);
        ctx.setTransform(s, 0, 0, s, (1 - s) * w / 2 + dx * w, (1 - s) * h);
        ctx.drawImage(this.dim.get(b).canvas, 0, 0);
        ctx.restore();
      }
    }

    // the real you, on top of every clone
    ctx.drawImage(this.cutout, 0, 0);
  }
}

/** Where each clone ends up on screen, from the bounding box of your cutout (frame coordinates). */
export function cloneBoxes(box, w, h, clones = CLONES) {
  if (!box) return [];
  return clones.map(([dx, s]) => {
    const ox = (1 - s) * w / 2 + dx * w;
    const oy = (1 - s) * h;
    return [s * box.x0 + ox, s * box.y0 + oy, s * box.x1 + ox, s * box.y1 + oy];
  });
}
