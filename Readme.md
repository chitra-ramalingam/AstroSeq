
Training On TESS  , Kepler , K2 , TOI Data
-----------------------
Kepler → IDs: KIC (e.g., KIC 7281229), time = BKJD (= BJD–2454833).

K2 (Kepler’s extended mission) → IDs: EPIC (e.g., EPIC 201367065), time = BKJD.

TESS → IDs: TIC (e.g., TIC 141914082) and often referenced via TOI numbers, time = BTJD (= BJD–2457000).

TESS/NEA columns: pl_tranmid (BJD_TDB), pl_orbper, pl_trandurh

Kepler/KOI columns: koi_time0bk (BKJD), koi_period, koi_duration

Matched the time systems to Lightkurve:

TESS light curves are BTJD →  converted pl_tranmid from BJD to BTJD 

Kepler/K2 light curves are BKJD → koi_time0bk is already in BKJD.

Labels: 
Get time & PDCSAP flux (quality mask, remove NaNs; optional outlier clip/flatten).
plit the flux into windows of length window with stride ≈ window/4, z-score each segment.
Kept a group id (TIC/KIC/EPIC/name) for every segment, capped how many segments we take per star (to prevent one star from dominating), and split train/test by group so the same star never leaks across splits.
Grouped splits prevents leakage


# Observations

Tried Isolation Forest and miserably failed.
Some of the flux were empty and too time to figure it out to remove them, got leaked in the training

So after so many fixes and retries got the following

Set Epochs from 20, 40, 60 and analysed.

# Observation at epoch 20
Good: PR-AUC is well above baseline (VAL 0.635 vs 0.493; TEST 0.678 vs 0.477) ⇒ the model’s ranking is useful.

Issue: on TEST threshold is too conservative → predicted positive rate ≈ 20% vs true positive rate ≈ 48%, giving recall 0.334.

# Observation at Epoch 40
Read on the plots/numbers

Overfitting: train loss ↓ while val loss ↑ steadily. Val AUC/PR-AUC hover ~0.60 ⇒ beyond ~15–20 epochs are not gaining generalization.

Generalization: TEST AUC 0.645 / PR-AUC 0.672 > VAL (0.604/0.612). That’s fine—split variance. 

# Observation at Epoch 60

Did nothing, as early stopping kicked in, and the it took Epoch as 20. Can set the patience to high 999 but ineffective i think.



