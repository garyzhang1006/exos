"""
Exos -- train a planet/not-a-planet classifier on NASA's real Kepler data.

Task: given a transit signal's physical features, predict whether it is a real
CONFIRMED exoplanet or a FALSE POSITIVE (eclipsing binary, noise, etc.) -- the
exact triage NASA scientists do. We deliberately EXCLUDE the koi_fpflag_* columns
(they encode the human false-positive verdict and would leak the label); the model
learns from physics only.

Outputs (into ../web/ for the browser app):
  web/model.onnx       gradient-boosting classifier, ONNX (runs via onnxruntime-web)
  web/artifacts.json   feature spec, scaler, interpretable logistic surrogate,
                       feature stats, metrics, and real example candidates
  model/metrics.txt    human-readable metrics

Run:  python model/train.py    (downloads data/koi.csv from NASA if missing)
"""
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WEB = ROOT / "web"; WEB.mkdir(exist_ok=True)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)
RANDOM_STATE = 42

# (koi column, human label, unit, teaching note, higher-means)
FEATURES = [
    ("koi_period",    "Orbital period",      "days",     "How long one orbit takes. Real planets span a huge range; very short periods need care."),
    ("koi_duration",  "Transit duration",    "hours",    "How long the star dims for. Must be consistent with the period and star size for a real planet."),
    ("koi_depth",     "Transit depth",       "ppm",      "How much the star's light drops. Deeper dip = bigger object crossing -- but a star-sized dip means an eclipsing binary, not a planet."),
    ("koi_prad",      "Planet radius",       "R_Earth",  "Inferred radius in Earth radii. Tens of Earth radii is star-sized -> usually a FALSE POSITIVE."),
    ("koi_teq",       "Equilibrium temp",    "K",        "Estimated temperature from the star's heat. Context for habitability, weak on its own."),
    ("koi_insol",     "Insolation flux",     "Earth=1",  "Starlight received vs Earth. Extreme values hint at very close or odd orbits."),
    ("koi_model_snr", "Signal-to-noise",     "",         "Strength of the transit signal. Low SNR often means noise or a marginal false positive."),
    ("koi_steff",     "Star temperature",    "K",        "Host star's surface temperature."),
    ("koi_slogg",     "Star surface gravity","log g",    "Host star's surface gravity; helps pin down the star's size."),
    ("koi_srad",      "Star radius",         "R_Sun",    "Host star's radius. Needed to turn a transit depth into a planet size."),
    ("koi_impact",    "Impact parameter",    "",         "How centrally the object crosses the star (0 = central). Grazing (>1) transits are often false positives."),
]
COLS = [f[0] for f in FEATURES]


def load_data():
    csv = DATA / "koi.csv"
    if not csv.exists():
        cols = ",".join(COLS + ["koi_disposition", "kepoi_name", "kepler_name"])
        url = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
               f"select+{cols}+from+cumulative&format=csv")
        print("downloading NASA Kepler cumulative KOI table...")
        urllib.request.urlretrieve(url, csv)
    return pd.read_csv(csv)


def main():
    df = load_data()
    print(f"loaded {len(df)} Kepler objects of interest")

    # binary task: CONFIRMED (1) vs FALSE POSITIVE (0); CANDIDATE rows are held out as "unknowns"
    labeled = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy()
    y = (labeled["koi_disposition"] == "CONFIRMED").astype(int).values
    X = labeled[COLS].astype(float)

    medians = X.median()
    X = X.fillna(medians).values
    print(f"training on {len(X)} labeled signals "
          f"({int(y.sum())} planets, {int((1-y).sum())} false positives)")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=RANDOM_STATE, stratify=y)

    # interpretable surrogate (standardized) -> drives the "why" panel
    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(Xtr), ytr)

    # shipped model: gradient boosting on raw features (trees don't need scaling)
    gbm = GradientBoostingClassifier(n_estimators=400, max_depth=3,
                                     learning_rate=0.05, subsample=0.9,
                                     random_state=RANDOM_STATE).fit(Xtr, ytr)

    def evaluate(name, model, Xte_, transform=None):
        Xt = transform(Xte_) if transform else Xte_
        p = model.predict_proba(Xt)[:, 1]
        pred = (p >= 0.5).astype(int)
        cvX = scaler.transform(X) if transform else X
        cv = cross_val_score(model, cvX, y, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
                             scoring="roc_auc")
        m = {"accuracy": float(accuracy_score(yte, pred)),
             "roc_auc": float(roc_auc_score(yte, p)),
             "f1": float(f1_score(yte, pred)),
             "cv_auc_mean": float(cv.mean()), "cv_auc_std": float(cv.std())}
        print(f"{name:20s} acc={m['accuracy']:.3f}  AUC={m['roc_auc']:.3f}  "
              f"F1={m['f1']:.3f}  CV-AUC={m['cv_auc_mean']:.3f}+/-{m['cv_auc_std']:.3f}")
        return m

    print("\n=== held-out 20% test ===")
    metrics = {
        "logistic": evaluate("logistic (surrogate)", logit, Xte, scaler.transform),
        "gradient_boosting": evaluate("gradient_boosting", gbm, Xte),
    }

    # feature stats over the full labeled set (for sliders + "vs dataset" bars)
    Xdf = pd.DataFrame(X, columns=COLS)
    feature_stats = {c: {"min": float(Xdf[c].min()), "max": float(Xdf[c].max()),
                         "mean": float(Xdf[c].mean()), "std": float(Xdf[c].std()),
                         "median": float(medians[c])} for c in COLS}

    # real example candidates for the chips: confirmed, false positives, and unknowns
    examples = pick_examples(df, medians)

    artifacts = {
        "dataset": {
            "name": "NASA Kepler Objects of Interest (cumulative)",
            "n": int(len(X)), "n_total": int(len(df)),
            "target": "CONFIRMED exoplanet vs FALSE POSITIVE",
            "source": "NASA Exoplanet Archive (exoplanetarchive.ipac.caltech.edu)",
            "citation": "NASA Exoplanet Archive, Kepler cumulative KOI table. Borucki et al. 2010.",
        },
        "features": [{"key": k, "label": l, "unit": u, "note": n} for (k, l, u, n) in FEATURES],
        "feature_order": COLS,
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "logistic": {"coef": logit.coef_[0].tolist(), "intercept": float(logit.intercept_[0])},
        "feature_stats": feature_stats,
        "medians": {c: float(medians[c]) for c in COLS},
        "metrics": metrics,
        "best_model": "gradient_boosting",
        "examples": examples,
    }
    (WEB / "artifacts.json").write_text(json.dumps(artifacts, indent=2))
    (HERE / "metrics.txt").write_text(
        "Exos -- Kepler planet classifier\n"
        f"dataset: {len(X)} labeled KOIs ({int(y.sum())} planets / {int((1-y).sum())} false positives)\n"
        + "\n".join(f"{k}: acc={v['accuracy']:.3f} AUC={v['roc_auc']:.3f} F1={v['f1']:.3f}"
                    for k, v in metrics.items()) + "\n"
    )

    export_onnx(gbm, len(COLS))
    print(f"\nwrote web/artifacts.json, web/model.onnx, model/metrics.txt")


def pick_examples(df, medians):
    """A handful of real, recognizable candidates across the three dispositions."""
    out = []
    seen = 0
    # prefer named confirmed planets (habitable-zone-ish), star-sized false positives, and unknowns
    def add(row, tag):
        feats = {c: (float(row[c]) if pd.notna(row[c]) else float(medians[c])) for c in COLS}
        name = row.get("kepler_name") if pd.notna(row.get("kepler_name")) else row.get("kepoi_name")
        out.append({"name": str(name), "disposition": row["koi_disposition"],
                    "tag": tag, "x": [round(feats[c], 4) for c in COLS]})

    famous = ["Kepler-22 b", "Kepler-186 f", "Kepler-452 b", "Kepler-62 f", "Kepler-69 c"]
    for nm in famous:
        m = df[df["kepler_name"] == nm]
        if len(m):
            add(m.iloc[0], "confirmed")
    # a couple of obvious false positives (huge radius -> star-sized)
    fp = df[(df["koi_disposition"] == "FALSE POSITIVE") & (df["koi_prad"] > 30)].head(2)
    for _, r in fp.iterrows():
        add(r, "false_positive")
    # a couple of still-unknown candidates for the "you decide" angle
    cand = df[df["koi_disposition"] == "CANDIDATE"].dropna(subset=["koi_prad"]).head(2)
    for _, r in cand.iterrows():
        add(r, "candidate")
    return out


def export_onnx(model, n_features):
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    onx = convert_sklearn(
        model, initial_types=[("input", FloatTensorType([None, n_features]))],
        options={id(model): {"zipmap": False}},  # plain probability tensor, not a dict
        target_opset=18,
    )
    (WEB / "model.onnx").write_bytes(onx.SerializeToString())


if __name__ == "__main__":
    main()
