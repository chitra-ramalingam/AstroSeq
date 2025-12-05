# AstroSeq: 1D CNN Transit Classifier for TESS / Kepler / K2 / TOI

AstroSeq is an end-to-end pipeline for **exoplanet transit detection** using **1D convolutional neural networks** on real mission data from **TESS**, **Kepler**, **K2**, and **TOIs**.  

It:

- Downloads light curves via **Lightkurve** (MAST backend).
- Builds **segment-level labels** from ephemerides (NASA Exoplanet Archive / KOI fields).
- Trains a **1D Inception-ResNet-style CNN** to classify segments as *transit / non-transit*.
- Aggregates segment scores to **star-level “transit likelihood” scores** (Starwise 1D CNN).

This repo is for learning, experimentation, and playing with exoplanet ML 
---

# How to run it
* Create .venv, activate venv

Run --> python main.py

## Data Sources & Time Systems

Supported missions / ID types:

- **Kepler**  
  - IDs: `KIC` (e.g. `KIC 7281229`)  
  - Time: **BKJD** = BJD − 2454833  
- **K2 (Kepler extended)**  
  - IDs: `EPIC` (e.g. `EPIC 201367065`)  
  - Time: **BKJD**  
- **TESS**  
  - IDs: `TIC` (e.g. `TIC 141914082`), often referenced via TOI numbers  
  - Time: **BTJD** = BJD − 2457000  

Ephemeris fields used:

- **TESS / NEA columns**
  - `pl_tranmid` (BJD_TDB) – transit midtime  
  - `pl_orbper` – orbital period (days)  
  - `pl_trandurh` – transit duration (hours)
- **Kepler / KOI columns**
  - `koi_time0bk` (BKJD) – transit midtime  
  - `koi_period` – orbital period (days)  
  - `koi_duration` – transit duration (hours)

Time system alignment with Lightkurve:

- **TESS light curves** are BTJD  
  → `pl_tranmid` (BJD) is converted to BTJD via `pl_tranmid - 2457000`.
- **Kepler/K2 light curves** are BKJD  
  → `koi_time0bk` is already in BKJD, used directly.

---

## Label Construction (Segment-Level)

For each star:

1. **Fetch & clean light curve**
   - Use Lightkurve to get PDCSAP flux.
   - Apply quality mask, remove NaNs.
   - Optional outlier clipping / flattening.

2. **Windowing**
   - Split the flux into overlapping segments:
     - window length = `window`
     - stride ≈ `window / 4`
   - For each segment, apply per-segment centering / z-score style normalization.

3. **Ephemeris-based labeling**
   - For each segment, check if its time span overlaps a predicted transit window (using midtime, period, and duration).
   - Assign **segment label**:
     - `1` if it overlaps a predicted transit,
     - `0` otherwise.

4. **Group handling**
   - Maintain a **group id** per segment (`TIC` / `KIC` / `EPIC` / hostname).
   - **Cap segments per star** (e.g. `per_star_cap`) so a single prolific star does not dominate the loss.
   - Use **grouped train/val/test splits** so the same star never appears in multiple splits → prevents leakage.

Grouped splits ensure **no star-level leakage** between train/validation/test.

---

## Model: 1D Inception-ResNet Style CNN

Core architecture (`build_inception_resnet_1d`):

- Input: shape `(window, channels)`
  - `channels` = number of features per timestep (e.g. 1 = flux, 2 = raw + flattened flux).
- Stem:
  - Conv + BatchNorm with 64 filters, kernel size 7 → captures local transit-like shapes (ingress, bottom, egress).
  - MaxPooling1D to downsample time.
- Inception-ResNet blocks:
  - Multiple Conv1D branches with different kernel sizes (multi-scale view).
  - Residual (skip) connections for stable training at depth.
  - A second stack of blocks after raising channel depth (e.g. to 96).
- Regularization:
  - SpatialDropout1D to drop entire feature maps and reduce overfitting.
- Head:
  - Global Average Pooling over time.
  - Dropout.
  - Dense(1, sigmoid) → **segment-level transit probability**.

Compiled with:

- Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
- Loss: `binary_crossentropy`
- Metrics:
  - ROC-AUC (`roc_auc`)
  - PR-AUC (`pr_auc`)
  - Binary accuracy

### NNet Intuition

- **Channels**: how many per-timestep features (flux variants).  
- **64 filters** early on: enough to learn a variety of transit and noise shapes without exploding model size.  
- **Kernel size 7**: wide enough to see the full shape of a dip, not just single spikes.

---

## Training & Metrics

Experiments across different epoch counts (with early stopping):

### Epoch 20

- **PR-AUC**:
  - Validation: ~0.635 (vs baseline ~0.493)
  - Test: ~0.678 (vs baseline ~0.477)
- Interpretation:
  - PR-AUC is well above baseline → **ranking is useful** (model pushes positives above negatives).
- Issue:
  - With a conservative threshold, predicted positive rate ≈ 20% vs true positive ≈ 48%.
  - **Segment recall ~0.33** → many positives are below the chosen threshold.

### Epoch 40

- Overfitting:
  - Train loss ↓ while val loss ↑.
  - Val ROC/PR linger around ~0.60 → little generalization gain beyond ~15–20 epochs.
- Generalization:
  - TEST ROC-AUC ≈ 0.645, PR-AUC ≈ 0.672 > VAL (~0.604 / 0.612).
  - Acceptable split variance.

### Epoch 60

- Early stopping kicks in around ~20 epochs.
- Increasing max epochs further has no practical effect without changing regularization or architecture.

### AUC Note

- **AUC is threshold-free and not a probability**.  
  It measures how often true positives score higher than true negatives. Good AUC means the **ranking** is meaningful, even if the raw scores are not probability-calibrated.

---

## Starwise 1D CNN: Star-Level Scoring

After training the segment-level CNN, a second stage produces **star-level scores**:

- **Astro1DCNN** (training stage)
  - Takes a limited number of positive and negative stars (e.g. ~20 each).
  - Caps segments per star.
  - Trains the segment model efficiently.

- **Starwise1DCnn** (inference / survey stage)
  - Reuses the trained model from `best_ref.keras`.
  - For each star:
    - Fetches full light curve (possibly multiple sectors/quarters).
    - Normalizes + segments **all** available data (no per-star cap).
    - Predicts per-segment probabilities with the trained CNN.
    - Aggregates segment scores into a **single star-level score**:
      - e.g. top-k mean, max, or similar.
  - Outputs:
    - `target_id`, `mission`, `num_segments`, `star_score`.

### Performance & Cost

- Starwise can be **computationally expensive**:
  - Astro1DCNN training sees a small, capped subset.
  - Starwise runs over all segments for all stars of interest.
  - Most of the wall-time is dominated by Lightkurve/MAST I/O, not the CNN.

- Caching:
  - Light curves are cached as `.npz` files.
  - Subsequent runs reuse local data (no repeated downloads).
  - Cache size can grow to tens of GB for large star lists.

---

## Isolation Forest Attempt

An earlier attempt used **Isolation Forest** for anomaly detection:

- Precision/recall tradeoffs were poor for this problem.
- Performance metrics hovered around:
  - Precision/recall ≈ 0.57–0.69
  - F1 ≈ 0.62
- It struggled with:
  - Highly imbalanced transit vs non-transit segments.
  - Complex variability patterns in real light curves.

Conclusion: Isolation Forest was not effective here; the 1D CNN approach is more suitable.

---

## Known Issues / Gotchas

- Some light curves are **empty, too short, or fail during download**:
  - These are recorded and cached as empty to avoid repeated fetch attempts.
  - Care is needed to avoid leaking problematic samples into training.
- **Download time**:
  - Fetching thousands of stars from Lightkurve/MAST is slow.
  - Caching is essential for practical experimentation.
- **Calibration**:
  - Segment probabilities are not fully calibrated.
  - Star-level scores are best interpreted as **relative ranking**, not physical probabilities.

  # Applied the star classifier to all stars.

Chose a threshold / top-N to define “positives”.
Saved 197 stars to CSV with columns:
star_id, score_raw , p_highclass (you stored 1 for those that passed your cut)
A model-selected set of strongly transit-like stars, containing a mix of confirmed planets, candidates, and likely false positives.

--- Chat GPT 5.1 helped me create this Readme.md

## Keywords (for discoverability)

`exoplanet`, `transit`, `TESS`, `Kepler`, `K2`, `TOI`, `Lightkurve`, `MAST`, `1D CNN`, `Inception ResNet`, `time series`, `astroinformatics`, `deep learning`, `astronomy`, `exoplanet detection`, `Python`, `TensorFlow`

---
