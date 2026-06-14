# Exos — *find planets the way NASA does* 🪐

When a star flickers, is it a real planet crossing in front of it — or a false alarm? **Exos** is a machine-learning model, trained on **NASA's real Kepler mission data**, that makes that call **in your browser**, draws the transit light-curve, and an **AI tutor (GPT-4o-mini)** teaches you the astronomy behind every prediction.

Built for **DSH Hacks V1** — *AI × STEM Education*.

> Pick a real Kepler object (or move the sliders) → a gradient-boosting model predicts **real planet vs. false positive** client-side → an interpretable model shows *which measurements drove the call* → an AI tutor explains the transit-method reasoning in plain English, grounded in the numbers.

---

## Why it fits "AI × STEM Education"

- **Real science, real data.** 9,564 Kepler Objects of Interest from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) — the same catalog astronomers use. The model learns the actual triage scientists perform.
- **It teaches, it doesn't just predict.** Every verdict is decomposed into per-measurement contributions, an animated transit light-curve makes the physics visible, and the AI tutor explains the *why* — grounded in this signal's real numbers, never inventing them.
- **Honest model.** We **exclude NASA's own false-positive flags** from the features, so the model learns from physics (orbital period, transit depth/duration, inferred planet size, signal strength, host star) — not from the answer key.
- **Broad audience.** Anyone who's looked up at the night sky — not a niche.

## What it does

| Feature | Description |
|---|---|
| **Live prediction** | Pick a real Kepler signal or nudge 11 sliders; get a planet-vs-false-positive probability, instantly, in-browser. |
| **Transit light-curve** | An animated star + planet + brightness dip that reacts to the depth and size you set. |
| **"Why?" panel** | Per-feature contributions from an interpretable logistic model — see what pushes the call toward planet or impostor. |
| **Truth check** | Famous presets (Kepler-22 b, 186 f, 452 b…) show whether the model agrees with NASA's real verdict. |
| **AI tutor** | Ask **GPT-4o-mini** *why* — it explains the transit method grounded in this signal's measurements. |

## Model & results

A gradient-boosting classifier on 11 physical features, **CONFIRMED vs FALSE POSITIVE** (7,586 labeled signals; the 1,978 still-unconfirmed CANDIDATEs are held out as "unknowns" you can test):

| Model | Test accuracy | ROC-AUC | F1 | CV-AUC |
|---|---|---|---|---|
| Logistic regression (interpretable surrogate) | 0.827 | 0.902 | 0.773 | 0.897 |
| **Gradient boosting (shipped)** | **0.932** | **0.980** | **0.907** | **0.975** |

## Architecture

```
Browser  ──  ONNX Runtime Web  ──▶  Kepler classifier (model.onnx)   ← prediction, 100% client-side
   │
   └── POST /api/tutor ──▶  server/server.py  ──▶  OpenAI GPT-4o-mini  ← tutor, streamed (key stays server-side)
```

The prediction needs no server. Only the AI tutor calls a backend, which holds the OpenAI key as an env var.

## Run it locally

```bash
# 1. Prediction-only (no key): any static server works
cd web && python3 -m http.server 8000     # open http://localhost:8000

# 2. With the AI tutor
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...               # never commit this
python server/server.py                    # serves web/ + the tutor at http://localhost:8000
```

## Deploy (Render)

1. Push to GitHub.
2. Render → **New → Blueprint** → pick this repo (uses `render.yaml`), or **New → Web Service** with build `pip install -r requirements.txt`, start `python server/server.py`.
3. Set **`OPENAI_API_KEY`** as an environment variable in the Render dashboard (secret — never in the repo). Render injects `$PORT` automatically.

> ⚠️ **Security:** the OpenAI key lives only in the host's environment. Never commit it; a key pushed to a public repo is scraped and drained within minutes. Rotate the key after the event.

## Retrain the model

```bash
pip install numpy pandas scikit-learn skl2onnx onnx onnxruntime
python model/train.py     # downloads NASA's KOI table, trains, exports web/model.onnx + web/artifacts.json
```

## Project layout

```
exos/
├── web/
│   ├── index.html        # the app (ONNX Runtime Web)
│   ├── app.css           # space-themed styles
│   ├── app.js            # in-browser prediction, transit animation, "why" panel, AI tutor
│   ├── model.onnx        # trained Kepler classifier
│   └── artifacts.json    # feature spec, scaler, logistic surrogate, stats, real examples
├── server/server.py      # OpenAI GPT-4o-mini tutor proxy (serves web/, streams /api/tutor)
├── model/train.py        # NASA data → features → train → export
├── render.yaml           # Render deploy blueprint
└── requirements.txt      # runtime dep: openai
```

## Credits

- **Data:** NASA Exoplanet Archive — Kepler cumulative KOI table. Borucki et al. 2010.
- **Tutor:** OpenAI GPT-4o-mini (explains; does not re-decide the verdict, and is never fine-tuned).
- ONNX Runtime Web · scikit-learn · skl2onnx.

Predictions are an educational model's estimates, not official NASA dispositions.

## License

MIT.
