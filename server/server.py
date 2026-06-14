#!/usr/bin/env python3
"""
Exos AI tutor backend — a thin proxy that keeps the OpenAI API key off the
browser and streams GPT-4o-mini explanations to the page.

The Kepler classifier (ONNX) runs 100% client-side and is unchanged. This server
only adds the optional tutor and serves the static app:

  GET  /                serves web/ (the app, model runs in the browser)
  POST /api/tutor       {prob, verdict, name, nasa_disposition, features[], question?}
                        -> Server-Sent-Events stream of the explanation

Run locally:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...        # never commit this
    python server/server.py            # http://localhost:8000

The model is gpt-4o-mini (the user's choice). We do NOT fine-tune it — it answers
questions, grounded in the trained classifier's output and this signal's real
measurements.
"""
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"
MODEL = "gpt-4o-mini"  # user-specified
PORT = int(os.environ.get("PORT", "8000"))

SYSTEM = """You are the tutor inside Exos, a tool that teaches how astronomers find exoplanets with NASA's Kepler data.

A machine-learning model (trained on real Kepler signals) has already decided whether THIS transit signal is a real planet or a false positive. Your job is to explain the astronomy and the model's reasoning so a curious student understands it — never to overturn the verdict.

Ground every claim in the numbers you are given:
- Tie the explanation to this signal's actual measurements (transit depth, inferred planet radius, orbital period, signal-to-noise, impact parameter, and the host star).
- Use the core ideas of the transit method: a planet blocks a tiny, repeating sliver of starlight; an eclipsing binary star blocks far more (a "planet" tens of Earth-radii wide is really a second star); low signal-to-noise or a grazing (high impact parameter) transit often means a false positive.
- If a real NASA disposition is provided, you may reference it, but explain the physics rather than just citing the label.

Style: warm, vivid, plain prose — no markdown headings or bullet lists. About 110-160 words unless the question needs more. Remind students this is an educational model estimate, not an official NASA determination, if they would over-trust it. If asked something off-topic, gently steer back to this signal."""


def build_user_message(p):
    verdict = p.get("verdict", "")
    prob = p.get("prob")
    pct = f"{round(float(prob)*100)}% chance it's a real planet" if prob is not None else ""
    lines = [
        f"The model says: {verdict} ({pct}).",
    ]
    if p.get("name"):
        nd = p.get("nasa_disposition")
        lines.append(f"This signal is {p['name']}" + (f", which NASA currently lists as {nd}." if nd else "."))
    lines.append("")
    lines.append("This signal's measurements (value — percentile across the dataset):")
    for f in p.get("features", []):
        unit = f.get("unit") or ""
        pctile = f.get("percentile")
        where = f"{pctile}th pct" if pctile is not None else "n/a"
        lines.append(f"  - {f.get('label')}: {f.get('value')} {unit} ({where})")
    q = (p.get("question") or "").strip()
    lines.append("")
    lines.append(f"Student's question: {q}" if q else
                 "Explain why the model reached this verdict — the transit-method reasoning behind it.")
    return "\n".join(lines)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, format, *args):
        sys.stderr.write("  %s\n" % (format % args))

    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.end_headers()

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _sse(self, obj):
        self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode()); self.wfile.flush()

    def do_POST(self):
        if self.path.rstrip("/") != "/api/tutor":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "invalid JSON"); return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self._cors(); self.end_headers()

        if not os.environ.get("OPENAI_API_KEY"):
            self._sse({"error": "OPENAI_API_KEY not set on the server. export it and restart."})
            return
        try:
            client = OpenAI()
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": build_user_message(payload)}],
                stream=True, temperature=0.4, max_tokens=500,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    self._sse({"text": delta})
            self._sse({"done": True})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # noqa: BLE001 — surface to the page
            self._sse({"error": f"{type(e).__name__}: {e}"})


def main():
    if not WEB.exists():
        sys.exit(f"web/ not found at {WEB}")
    key = "set" if os.environ.get("OPENAI_API_KEY") else "MISSING (tutor will error)"
    print(f"Exos tutor on http://localhost:{PORT}  model={MODEL}  OPENAI_API_KEY={key}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
