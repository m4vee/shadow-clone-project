// Demo page wiring. The reusable part is ShadowCloneEngine; this file is only the UI around it.
import { ShadowCloneEngine } from './shadow-clone.js';

const hide = () => { img.closest('.side-box').hidden = true; };

const $ = (id) => document.getElementById(id);
const statusEl = $('status');
const setStatus = (msg, isError = false) => {
  statusEl.textContent = msg;
  statusEl.classList.toggle('error', isError);
};

const engine = new ShadowCloneEngine({
  canvas: $('stage'),
  video: $('cam'),
  onStatus: (m) => setStatus(m),
});

const buttons = ['stop', 'spawn', 'dismiss', 'debug'].map($);
const setRunning = (on) => {
  buttons.forEach((b) => (b.disabled = !on));
  $('start').disabled = false;
  $('placeholder').hidden = on;
};

function friendlyError(err) {
  if (err && (err.name === 'NotAllowedError' || err.name === 'SecurityError')) {
    return 'Camera permission was blocked. Allow the camera for this site in your browser, then press Start again.';
  }
  if (err && err.name === 'NotFoundError') return 'No camera found on this device.';
  if (err && err.name === 'NotReadableError') return 'The camera is busy. Close other apps using it (Zoom, Teams, other tabs) and try again.';
  if (!window.isSecureContext) return 'Camera access needs https:// or http://localhost.';
  return `Could not start: ${(err && err.message) || err}. If this mentions loading models, check your internet connection.`;
}

$('start').addEventListener('click', async () => {
  $('start').disabled = true;
  try {
    await engine.start();
    setRunning(true);
  } catch (err) {
    console.error(err);
    setStatus(friendlyError(err), true);
    setRunning(false);
  }
});

$('stop').addEventListener('click', () => {
  engine.stop();
  setRunning(false);
});
$('spawn').addEventListener('click', () => engine.spawn());
$('dismiss').addEventListener('click', () => engine.dismiss());
$('debug').addEventListener('click', () => {
  $('debug').setAttribute('aria-pressed', String(engine.toggleDebug()));
});

const thr = $('thr');
thr.addEventListener('input', () => {
  engine.setThreshold(parseFloat(thr.value));
  $('thrOut').textContent = parseFloat(thr.value).toFixed(2);
});
const lead = $('lead');
lead.addEventListener('input', () => {
  engine.setSpawnSoundLead(parseInt(lead.value, 10) / 1000);
  $('leadOut').textContent = `${lead.value} ms`;
});

function nudgeThreshold(delta) {
  engine.setThreshold(engine.threshold + delta);
  thr.value = engine.threshold.toFixed(2);
  $('thrOut').textContent = engine.threshold.toFixed(2);
}

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement || e.metaKey || e.ctrlKey || e.altKey) return;
  if (!engine.running) return;
  switch (e.key.toLowerCase()) {
    case 's': engine.spawn(); break;
    case 'c': engine.dismiss(); break;
    case 'd': $('debug').setAttribute('aria-pressed', String(engine.toggleDebug())); break;
    case 'i': engine.toggleInvert(); break;
    case '[': nudgeThreshold(-0.05); break;
    case ']': nudgeThreshold(0.05); break;
  }
});

window.addEventListener('pagehide', () => engine.destroy());
