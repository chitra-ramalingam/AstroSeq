
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