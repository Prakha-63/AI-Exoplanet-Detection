# NASA NEO Hazard Classification — a leakage-aware redo

Predicting whether a Near-Earth Object is **potentially hazardous (PHA)** — done honestly.
This project started as a beginner EDA and was rebuilt into a rigorous ML pipeline whose
headline result is not a leaderboard number but a **diagnosis**: the standard dataset can only
get you so far because it's missing half the physics of the label — and then a fix that adds
that physics back and breaks the ceiling for the *right* reason.

> **TL;DR** — Splitting the data the way most public notebooks do leaks asteroid identity and
> inflates the score. Split correctly and the model caps at **PR-AUC ≈ 0.50**, because the
> `hazardous` label depends on **Earth MOID**, which isn't in the dataset. Pull real orbital
> elements from JPL's SBDB and add MOID, and PR-AUC jumps to **≈ 0.99** — not because the model
> got smarter, but because the information gap was closed.

---

## The core idea

NASA's **Potentially Hazardous Asteroid** flag is defined by two conditions:

```
PHA  ⇔  Earth MOID ≤ 0.05 AU   AND   absolute magnitude H ≤ 22
```

- **H** (a size proxy) *is* in the Kaggle dataset.
- **MOID** (Minimum Orbit Intersection Distance, an orbital property) *is not*.

So the label is only **half-learnable** from the data as shipped. Any model trained on it can
learn the `H ≤ 22` gate exactly and is then forced to guess the MOID half from noisy
per-approach proxies (velocity, miss-distance) that only loosely correlate with it. This project
makes that limitation explicit, proves it with explainability, and then fixes it with real data.

## What the notebook does

| Step | What | Why it matters |
|------|------|----------------|
| **1. Data-integrity audit** | 90,836 rows → 27,423 unique objects (~3.3 close-approach rows each); `est_diameter_*` shown to be an exact function of H; class balance 9.7% | Finds the traps *before* modeling |
| **2. Label analysis** | `P(hazardous \| H ≤ 22) = 30%` only — the MOID half is missing | Establishes the information ceiling |
| **3. Two eval protocols** | Naive row-split **PR-AUC 0.53** vs. group-split on `id` **PR-AUC 0.50** | The gap *is* identity leakage; PR-AUC is primary, accuracy is rejected on a 10%-positive problem |
| **4. Explainability (SHAP)** | The model just re-learns the `H = 22` gate | Confirms it's an information ceiling, not a modeling failure |
| **5. Add the physics (JPL SBDB)** | Keyless bulk pull of Earth MOID + orbital elements for 42k NEOs | The missing feature, from the authoritative source |
| **6. Break the ceiling** | base **PR-AUC 0.69** → + MOID **0.99** | Fixed with *data*, not a bigger model |
| **7. Calibration** | `CalibratedClassifierCV` (isotonic) improves Brier score | Turns scores into usable risk probabilities |

## Key results

```
Protocol A (naive row split, leaky):   PR-AUC = 0.53   ← inflated by identity leakage
Protocol B (grouped split on id):      PR-AUC = 0.50   ← the honest, deployable number

Object-level, base features (H + approach aggregates):   PR-AUC = 0.69
Object-level, base + Earth MOID:                         PR-AUC = 0.99  ← ceiling broken
```

The near-perfect score with MOID is **not** the model being clever — MOID + H literally *is* the
label's definition, so adding MOID closes the information gap and the task nearly reduces to
recomputing a rule. The small residual gap is itself meaningful: SBDB's MOID comes from *today's*
orbit solution while the Kaggle labels were assigned years ago, so objects near the 0.05 AU
boundary drift across it — exactly the label-vs-feature epoch mismatch a deployed system faces.

## Methodology notes (the honest-engineering bits)

- **Grouped cross-validation.** With multiple close-approach rows per asteroid, a row-level split
  leaks object identity into the test set. All object-level evaluation uses one row per object;
  the row-level comparison uses `StratifiedGroupKFold` on `id` to quantify the leakage.
- **PR-AUC over accuracy.** At ~10% positive rate, "predict all-negative" scores 90% accuracy.
  Average precision is the primary metric throughout.
- **Dropped `est_diameter_*`.** It's a deterministic function of H (verified: `log10(D) + H/5` is
  constant to 1e-10) — zero new information, pure collinearity.
- **The SPK-ID join gotcha.** NASA revised the asteroid SPK-ID scheme in 2021, so the Kaggle `id`
  no longer equals SBDB's current `spkid` — a naive numeric join matches only ~23% of objects.
  The notebook joins on **primary designation** instead, recovering **96.8%** with prevalence
  preserved (no selection bias).

## Data sources

- **NASA NEO dataset** — Kaggle: [`sameepvani/nasa-nearest-earth-objects`](https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects) (auto-downloaded via `kagglehub`).
- **Orbital elements + Earth MOID** — [JPL SSD/CNEOS SBDB Query API](https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html) (keyless; pulled and cached to `sbdb_neo_elements.csv`).
- **Live NEO feed** (`Fetch_data.ipynb`) — [NASA NeoWs](https://api.nasa.gov/) (requires a free personal key).

## Repository layout

```
NEO_Hazard_Classification_upgraded.ipynb   ← main deliverable (run top-to-bottom)
Nasa NEO Analysis.ipynb                     ← original beginner EDA, kept as the "before" baseline
Fetch_data.ipynb                            ← optional: live NeoWs pull (needs NASA_API_KEY)
sbdb_neo_elements.csv                       ← cached JPL SBDB pull (lets the notebook run offline)
neo_dataset.csv                             ← small sample fetched by Fetch_data.ipynb
requirements.txt
```

## Setup & run

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

jupyter notebook NEO_Hazard_Classification_upgraded.ipynb   # run all cells
```

The upgraded notebook needs no API key — it uses Kaggle (via `kagglehub`) and the keyless JPL
SBDB API. Only `Fetch_data.ipynb` (the optional live NeoWs feed) needs a key:

```bash
# Get a free key at https://api.nasa.gov, then:
export NASA_API_KEY="your-key-here"     # PowerShell:  $env:NASA_API_KEY = "your-key"
```

**Never hardcode the key in a notebook or commit it.**

## Tech stack

Python 3.13 · pandas · scikit-learn · XGBoost · SHAP · matplotlib · kagglehub · JPL SBDB API
