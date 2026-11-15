# Shadow Clone Jutsu (web)

Make the Shadow Clone hand sign at your webcam and clones of you appear behind you, with smoke and sound. Make the dismiss sign and they vanish. This is the browser port of the Python version: same detector thresholds, same clone layout, same smoke.

Everything runs client-side (MediaPipe Tasks Vision in the visitor's browser). No backend, no video leaves the device.

## Run the demo

Camera access needs `https://` or `http://localhost`, so don't open `index.html` by double-clicking it. From this folder:

```
python -m http.server 8000
```

Then open http://localhost:8000 and press **Start camera**.

Put your sound files in `assets/` with these exact names (the page runs silent without them):

- `assets/shadow_clone_sound.mp3` plays when the clones appear
- `assets/shadow_clone_poof.mp3` plays when they vanish

## Add it to your own web app

Copy the `src/` folder (everything except `main.js`, which is only the demo UI) and use the engine:

```html
<canvas id="stage"></canvas>
<button id="go">Start</button>

<script type="module">
  import { ShadowCloneEngine } from './src/shadow-clone.js';

  const engine = new ShadowCloneEngine({
    canvas: document.getElementById('stage'),
    sounds: { spawn: '/audio/spawn.mp3', dismiss: '/audio/poof.mp3' },
    onEvent: (e) => console.log(e.type),      // 'spawn' | 'dismiss'
    onStatus: (msg) => console.log(msg),
  });

  // start() must run inside a click/tap so the browser allows camera + audio
  document.getElementById('go').onclick = () => engine.start();
</script>
```

In React, Vue, Svelte and similar frameworks, create the engine once (in `useEffect`/`onMounted`), pass it a canvas ref, call `engine.start()` from a button handler, and call `engine.destroy()` on unmount.

| Method | What it does |
| --- | --- |
| `start()` / `stop()` | Open or close the camera and loop. `start()` loads models on first use. |
| `spawn()` / `dismiss()` | Trigger the effect without a hand sign. |
| `setThreshold(0.3–0.95)` | Person-cutout strictness (same as `[` / `]` in the Python version). |
| `setSpawnSoundLead(seconds)` | Start the spawn sound this much earlier than the clones. |
| `toggleDebug()` | Hand skeletons and progress bars. |
| `destroy()` | Free models, camera and audio. |

Options worth knowing: `signConfig` overrides any threshold in `src/sign.js` (for example `{ HOLD_SECONDS: 0.4 }`), `trimSilence: true` skips leading silence in the mp3s, and `width`/`height` set the internal working size (default 960x540).

## Before you ship

- **Self-host the MediaPipe files.** By default the page loads the library and wasm from jsDelivr (`@mediapipe/tasks-vision@0.10.14`) and the two models from `storage.googleapis.com`. For production, download them, serve them from your own domain, and set `visionUrl`, `wasmUrl`, `handModelUrl` and `segModelUrl`.
- **Serve over https.** Browsers only allow camera access on secure origins (localhost excepted).
- **Check your sound files' licences.** The mp3s are yours to supply, and sound effects ripped from the anime may not be cleared for a public site.
- **Naruto and Shadow Clone Jutsu are Kishimoto / Shueisha properties.** Fine for a personal project; for a public or commercial site, use your own naming and assets.

## Troubleshooting

- **Cutout is the background, not you:** press `I` to flip the mask (it also auto-flips after about a second if the corners are solid and the centre is empty).
- **Chair or wall bits stay in the cutout:** raise the threshold slider, and add light on your face and torso.
- **Spawn sound feels late or early:** move the "Spawn sound lead" slider. Higher plays it earlier.
- **Low frame rate:** lower `width`/`height` (for example 640x360). The GPU path is used when the browser supports it.
- **Detection feels off in a different room or camera:** open the debug overlay and compare with the numbers you tuned in Python; thresholds are in `src/sign.js`.
