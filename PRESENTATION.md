# Exos — Submission Kit (DSH Hacks V1)

Everything you need to submit + present. Copy/paste the relevant blocks.

---

## 1. Elevator pitch (one line)

> A machine-learning model trained on NASA's real Kepler data tells you if a star's flicker is a true planet or a false alarm — in your browser — and an AI tutor teaches you the astronomy behind every call.

---

## 2. Demo video script (~75 seconds — read aloud while you screen-record)

> **[0:00 — hook, show the hero]**
> "When NASA's Kepler telescope watches a star dim, it could be a planet crossing in front of it — or just two stars eclipsing each other. Telling them apart is real, hard science. This is **Exos** — it does it in your browser."
>
> **[0:12 — click the *Kepler-22 b* chip]**
> "I'll pick Kepler-22 b, a real confirmed planet. A gradient-boosting model — trained on 7,586 real Kepler signals — runs **right here in the browser** and says: 93% likely a real planet. And see this line? That's NASA's actual verdict: confirmed. The model agrees."
>
> **[0:28 — drag the *Planet radius* slider way up]**
> "Now watch what happens if I make the planet huge — tens of Earth-radii. The verdict flips to **false positive**. Why? An object that big isn't a planet; it's a second star. The light-curve dip gets enormous. The model learned that physics from real data."
>
> **[0:44 — scroll to the *Why* panel]**
> "It's not a black box. This panel shows exactly which measurements pushed the call — transit depth, planet size, signal strength."
>
> **[0:52 — click *Explain this verdict* on the tutor]**
> "And if you don't know the science, the AI tutor — GPT-4o-mini — explains it in plain English, grounded in this signal's actual numbers. It teaches the transit method, not just the answer."
>
> **[1:05 — close]**
> "Real NASA data, a real model running in your browser, and an AI that teaches the astronomy behind it. That's Exos — learning to find planets the way NASA does."

**Recording tips:** full-screen the app, cursor visible, do the slider drag slowly so the verdict flip and the light-curve change are obvious. ~75s is ideal; DSH wants a clear demo of purpose + features + interaction.

---

## 3. Devpost "About the project" (paste into the big Markdown box)

```markdown
## Inspiration

NASA's Kepler telescope watched 150,000 stars for years, hunting for the faint, repeating dip in brightness a planet makes when it crosses its star. But most dips aren't planets — they're eclipsing binary stars, noise, or instrument quirks. Sorting the real planets from the false alarms is genuine, hard astronomy. We wanted to let any student *do that science* — and actually understand it — instead of reading about it.

## What it does

You pick a real Kepler signal (or move 11 sliders), and in real time, in your browser, Exos:
- **Predicts** whether the signal is a real planet or a false positive, with a confidence, using a model trained on NASA's actual catalog.
- **Draws** an animated transit light-curve — the star, the crossing planet, and the dip in brightness — that reacts to what you set.
- **Explains "why"** with an interpretable model that shows which measurements pushed the verdict toward planet or impostor.
- **Checks against NASA** — famous presets like Kepler-22 b show whether the model agrees with the real, published disposition.
- **Tutors you** — GPT-4o-mini explains the transit-method reasoning in plain English, grounded in this signal's real numbers. It explains; it never invents the verdict.

## How we built it

A real machine-learning pipeline, trained offline and shipped to the browser:
- **Data:** 9,564 Kepler Objects of Interest from the NASA Exoplanet Archive (cumulative KOI table).
- **Features:** 11 physical measurements — orbital period, transit depth and duration, inferred planet radius, equilibrium temperature, insolation, signal-to-noise, impact parameter, and the host star's temperature, gravity, and radius. We deliberately **exclude NASA's own false-positive flags**, so the model learns from physics, not the answer key.
- **Model:** a gradient-boosting classifier (scikit-learn) trained on CONFIRMED vs FALSE POSITIVE, exported to **ONNX** and run client-side with ONNX Runtime Web.
- **Tutor:** a tiny Python proxy holds the OpenAI key server-side and streams **GPT-4o-mini**, grounded on the model's output and the signal's measurements. No fine-tuning.

## Challenges we ran into

The biggest trap was **label leakage**: the dataset ships with NASA's own false-positive flags, and a model that sees them just memorizes the answer and learns no science. We removed them and trained on physics alone — the model still hits 0.98 ROC-AUC, which means the physical features genuinely separate planets from impostors. The second challenge was keeping the AI **honest**: an LLM that invents solubility-style numbers would defeat the educational point, so the tutor is given the model's verdict and the real measurements and told to explain, not re-decide.

## Accomplishments that we're proud of

- A model that reaches **93.2% accuracy and 0.980 ROC-AUC** on held-out real Kepler data — learning from physics, not NASA's answer key.
- The whole prediction runs **100% in the browser** — no server needed to classify.
- An animated transit light-curve that makes an abstract method visible.
- A grounded AI tutor that teaches the astronomy instead of hand-waving.

## What we learned

- How real exoplanet vetting works — why a huge inferred radius means "second star," why a grazing transit or low signal-to-noise screams false positive.
- Why excluding leaky features matters, and how to prove a model learned the science (it still works without the answer key).
- That the right role for an LLM in education is to **explain a trustworthy model**, grounded in real data — not to be the model.

## What's next for Exos

- Add TESS (the newer planet-hunting mission) data for fresh candidates.
- Let students upload a raw light-curve and detect the transit themselves.
- A "confidence over time" view showing how a candidate's odds change as more data arrives.
- Classroom mode: a guided lesson sequence built on the tutor.
```

---

## 4. Built with (tags)

```
python, javascript, scikit-learn, onnx, skl2onnx, onnx-runtime-web, openai, gpt-4o-mini, nasa-exoplanet-archive, html, css, render
```

## 5. "Try it out" links

```
https://<your-render-app>.onrender.com      ← live demo (fill in after deploy)
https://github.com/garyzhang1006/exos        ← code
```

---

## 6. One-page PDF (paste into a doc, export PDF)

**Exos — find planets the way NASA does**
*DSH Hacks V1 · AI × STEM Education*

**Problem.** NASA's Kepler mission found thousands of "transit" signals — tiny dips in starlight — but most are false alarms (eclipsing binary stars, noise), not planets. Telling them apart is real astronomy that students rarely get to *do*.

**Solution.** Exos is a machine-learning model trained on NASA's real Kepler catalog (9,564 objects) that classifies a signal as a real planet or a false positive **in the browser**, visualizes the transit light-curve, explains which measurements drove the call, and — via an AI tutor (GPT-4o-mini) — teaches the astronomy behind it in plain English.

**How it works.** A gradient-boosting classifier on 11 physical features (period, transit depth/duration, planet radius, signal-to-noise, host-star properties…), exported to ONNX and run client-side. NASA's own false-positive flags are excluded so the model learns from physics, not the answer key — and still reaches **93.2% accuracy, 0.980 ROC-AUC**. The AI tutor is grounded in the model's output and the real numbers; it explains, never re-decides.

**Why it matters for STEM education.** Students don't just read about exoplanet detection — they run the real classifier, flip a verdict by changing a measurement, see the physics animate, and get a tutor that teaches the *why*. Real data, a real model, real understanding.

**Links.** Live demo: `<render url>` · Code: github.com/garyzhang1006/exos
```
