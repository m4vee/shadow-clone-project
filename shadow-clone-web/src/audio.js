/**
 * Low-latency sound effects using the Web Audio API.
 *
 * Sounds are fetched and decoded once up front, so playing one is instant
 * (loading an <audio> element at play time is what makes effects feel late).
 * Browsers only allow audio after a user gesture, so call init() from a click.
 */
export class SoundBank {
  constructor() {
    this.ctx = null;
    this.sounds = new Map();
  }

  /** Call synchronously from a click/tap handler. */
  init() {
    if (!this.ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) this.ctx = new AC({ latencyHint: 'interactive' });
    }
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
  }

  /**
   * Fetch + decode a sound. Never throws: a missing file just means silence.
   * @param {string} name
   * @param {string} url
   * @param {{trimSilence?: boolean}} opts  skip leading silence in the file
   * @returns {Promise<boolean>} whether it loaded
   */
  async load(name, url, { trimSilence = false } = {}) {
    if (!this.ctx) return false;
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const buffer = await this.ctx.decodeAudioData(await res.arrayBuffer());
      let offset = 0;
      if (trimSilence) {
        const data = buffer.getChannelData(0);
        let i = 0;
        while (i < data.length && Math.abs(data[i]) < 0.01) i++;
        offset = Math.max(0, i / buffer.sampleRate - 0.005);
      }
      this.sounds.set(name, { buffer, offset });
      return true;
    } catch (err) {
      console.warn(`Sound "${name}" not loaded from ${url}:`, err.message || err);
      return false;
    }
  }

  play(name) {
    const s = this.sounds.get(name);
    if (!s || !this.ctx) return;
    if (this.ctx.state === 'suspended') this.ctx.resume();
    const src = this.ctx.createBufferSource();
    src.buffer = s.buffer;
    src.connect(this.ctx.destination);
    src.start(0, s.offset);
  }

  has(name) {
    return this.sounds.has(name);
  }

  close() {
    this.sounds.clear();
    if (this.ctx) this.ctx.close();
    this.ctx = null;
  }
}
