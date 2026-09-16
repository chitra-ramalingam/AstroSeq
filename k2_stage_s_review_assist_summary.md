# Stage S Review-Assist Summary

Date: 2026-04-22

Scope: descriptive review assist only. This does not fill `reviewer_outcome`, does not set `promote_to_deeper_eval`, does not replace the default policy, and does not widen Stage O beyond the 12 selected rows.

## Inputs

- Manual review sheet: `k2_stage_r_manual_review_sheet.csv`
- Event tables: each row's `events_csv`
- Available EPIC artifacts: each row's `epic_dir`
- Review-assist notes: `k2_stage_s_review_assist_notes.csv`

## Read This First

These notes are intentionally descriptive. They point to event-table patterns that look coherent or suspicious, but they do not make the final promotion, hold, or rejection call.

Every row remains a Stage O `review_high` case with severe autocorrelation and whiteness underflow risk. Use these notes to decide what to inspect first, not to bypass manual review.

## High-Score Bucket

### EPIC_211529255

Stage O score `29.895`, `n_events=18`, best depth SNR `38.721`, best shape `0.769`.

18 event rows across 71.0 d; depth_snr median/max 8.7/38.7; shape median/max 0.68/0.77; duration median/range 2/2-6 cadences. Top events: t=2345.574 snr=38.7 shape=0.68 dur=3; t=2348.843 snr=28.8 shape=0.71 dur=2; t=2344.655 snr=24.5 shape=0.71 dur=2.

Coherence hint: Mixed but usable morphology signal: median shape 0.68 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 71.0 d with median gap 1.08 d and a large max gap 30.9 d; check whether timing is episodic rather than periodic.

Review focus: Start by checking whether the many short events are true repeated dips or cadence/single-point artifacts. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 11/18 events are <=2 cadences, so single/cadence-scale dips need close artifact review.

### EPIC_212007631

Stage O score `29.817`, `n_events=44`, best depth SNR `37.616`, best shape `0.753`.

44 event rows across 72.5 d; depth_snr median/max 13.0/37.6; shape median/max 0.66/0.75; duration median/range 3/2-7 cadences. Top events: t=2329.862 snr=37.6 shape=0.68 dur=3; t=2350.886 snr=33.2 shape=0.71 dur=2; t=2345.738 snr=29.4 shape=0.71 dur=2.

Coherence hint: Mixed but usable morphology signal: median shape 0.66 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 72.5 d with median gap 0.84 d and a large max gap 11.3 d; check whether timing is episodic rather than periodic.

Review focus: Start by checking whether the many short events are true repeated dips or cadence/single-point artifacts. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 20/44 events are <=2 cadences, so single/cadence-scale dips need close artifact review.

### EPIC_211805106

Stage O score `29.797`, `n_events=44`, best depth SNR `37.832`, best shape `0.713`.

44 event rows across 72.4 d; depth_snr median/max 13.7/37.8; shape median/max 0.67/0.71; duration median/range 2/2-5 cadences. Top events: t=2326.695 snr=37.8 shape=0.68 dur=4; t=2322.752 snr=32.6 shape=0.68 dur=3; t=2324.734 snr=31.9 shape=0.71 dur=2.

Coherence hint: Mixed but usable morphology signal: median shape 0.67 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 72.4 d with median gap 1.08 d, but 4 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start by checking whether the many short events are true repeated dips or cadence/single-point artifacts. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 24/44 events are <=2 cadences, so single/cadence-scale dips need close artifact review. 4 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_211759736

Stage O score `29.669`, `n_events=29`, best depth SNR `194.977`, best shape `0.720`.

29 event rows across 71.2 d; depth_snr median/max 12.4/195.0; shape median/max 0.63/0.72; duration median/range 3/2-10 cadences. Top events: t=2335.848 snr=195.0 shape=0.70 dur=9; t=2372.421 snr=187.3 shape=0.70 dur=10; t=2327.410 snr=43.6 shape=0.66 dur=7.

Coherence hint: Morphology is mixed: median shape 0.63 and shape range 0.48-0.72; use event table to separate strongest events from weak tails. Timing hint: Events span 71.2 d with median gap 1.04 d and a large max gap 14.2 d; check whether timing is episodic rather than periodic.

Review focus: Start with the top-SNR events, then verify whether lower-SNR events repeat the same morphology and timing pattern. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high.

## Median-Score Bucket

### EPIC_211955365

Stage O score `28.477`, `n_events=24`, best depth SNR `105.420`, best shape `0.797`.

24 event rows across 73.3 d; depth_snr median/max 10.2/105.4; shape median/max 0.72/0.80; duration median/range 18/2-24 cadences. Top events: t=2378.265 snr=105.4 shape=0.75 dur=24; t=2371.440 snr=101.1 shape=0.72 dur=24; t=2368.151 snr=68.1 shape=0.80 dur=22.

Coherence hint: Event-table morphology looks comparatively coherent: median shape 0.72, all 24 rows have ingress_egress_ok=True. Still verify visually. Timing hint: Events span 73.3 d with median gap 3.35 d and max gap 6.6 d; spacing is worth checking for repeated structure, not assumed periodic.

Review focus: Start by checking whether the long high-SNR events are repeated astrophysical-like events or broad K2/systematic depressions. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 11/24 events are >=20 cadences, so broad trend/systematic explanations need close review.

### EPIC_211537297

Stage O score `28.468`, `n_events=35`, best depth SNR `79.723`, best shape `0.798`.

35 event rows across 71.2 d; depth_snr median/max 7.2/79.7; shape median/max 0.67/0.80; duration median/range 3/2-22 cadences. Top events: t=2376.304 snr=79.7 shape=0.79 dur=22; t=2372.708 snr=39.7 shape=0.76 dur=17; t=2346.473 snr=21.4 shape=0.67 dur=3.

Coherence hint: Mixed but usable morphology signal: median shape 0.67 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 71.2 d with median gap 1.27 d, but 6 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start with the top-SNR events, then verify whether lower-SNR events repeat the same morphology and timing pattern. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 6 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_211525754

Stage O score `28.461`, `n_events=23`, best depth SNR `111.398`, best shape `0.807`.

23 event rows across 66.2 d; depth_snr median/max 12.7/111.4; shape median/max 0.68/0.81; duration median/range 16/2-24 cadences. Top events: t=2368.438 snr=111.4 shape=0.81 dur=24; t=2348.619 snr=96.5 shape=0.75 dur=24; t=2375.058 snr=89.4 shape=0.75 dur=23.

Coherence hint: Mixed but usable morphology signal: median shape 0.68 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 66.2 d with median gap 3.05 d, but 4 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start by checking whether the long high-SNR events are repeated astrophysical-like events or broad K2/systematic depressions. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 11/23 events are >=20 cadences, so broad trend/systematic explanations need close review. 4 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_211421186

Stage O score `28.457`, `n_events=25`, best depth SNR `98.431`, best shape `0.765`.

25 event rows across 73.6 d; depth_snr median/max 9.2/98.4; shape median/max 0.65/0.77; duration median/range 2/2-23 cadences. Top events: t=2315.111 snr=98.4 shape=0.70 dur=23; t=2345.084 snr=56.6 shape=0.73 dur=2; t=2349.354 snr=23.1 shape=0.70 dur=2.

Coherence hint: Mixed but usable morphology signal: median shape 0.65 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 73.6 d with median gap 3.02 d, but 4 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start by checking whether the many short events are true repeated dips or cadence/single-point artifacts. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 13/25 events are <=2 cadences, so single/cadence-scale dips need close artifact review. 4 close event pairs under 0.25 d raise clustering/systematic concerns.

## Low-Score Bucket

### EPIC_211530033

Stage O score `27.399`, `n_events=26`, best depth SNR `181.628`, best shape `0.809`.

26 event rows across 73.5 d; depth_snr median/max 11.5/181.6; shape median/max 0.72/0.81; duration median/range 17/2-24 cadences. Top events: t=2349.537 snr=181.6 shape=0.75 dur=24; t=2333.662 snr=119.8 shape=0.79 dur=24; t=2318.114 snr=106.5 shape=0.81 dur=24.

Coherence hint: Event-table morphology looks comparatively coherent: median shape 0.72, all 26 rows have ingress_egress_ok=True. Still verify visually. Timing hint: Events span 73.5 d with median gap 3.31 d, but 5 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start with the top-SNR events, then verify whether lower-SNR events repeat the same morphology and timing pattern. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 5 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_212003686

Stage O score `27.310`, `n_events=24`, best depth SNR `359.525`, best shape `0.843`.

24 event rows across 73.6 d; depth_snr median/max 33.9/359.5; shape median/max 0.69/0.84; duration median/range 21/2-24 cadences. Top events: t=2317.664 snr=359.5 shape=0.69 dur=24; t=2313.557 snr=332.3 shape=0.72 dur=23; t=2326.041 snr=274.6 shape=0.74 dur=24.

Coherence hint: Mixed but usable morphology signal: median shape 0.69 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 73.6 d with median gap 4.09 d, but 4 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start by checking whether the long high-SNR events are repeated astrophysical-like events or broad K2/systematic depressions. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 15/24 events are >=20 cadences, so broad trend/systematic explanations need close review. 4 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_211972767

Stage O score `27.037`, `n_events=33`, best depth SNR `385.129`, best shape `0.817`.

33 event rows across 73.7 d; depth_snr median/max 11.8/385.1; shape median/max 0.71/0.82; duration median/range 17/2-24 cadences. Top events: t=2373.565 snr=385.1 shape=0.78 dur=22; t=2330.781 snr=265.3 shape=0.70 dur=20; t=2347.923 snr=261.4 shape=0.71 dur=23.

Coherence hint: Event-table morphology looks comparatively coherent: median shape 0.71, all 33 rows have ingress_egress_ok=True. Still verify visually. Timing hint: Events span 73.7 d with median gap 2.61 d, but 6 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start with the top-SNR events, then verify whether lower-SNR events repeat the same morphology and timing pattern. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 6 close event pairs under 0.25 d raise clustering/systematic concerns.

### EPIC_211490307

Stage O score `27.037`, `n_events=31`, best depth SNR `173.332`, best shape `0.819`.

31 event rows across 74.8 d; depth_snr median/max 13.0/173.3; shape median/max 0.69/0.82; duration median/range 11/2-24 cadences. Top events: t=2347.312 snr=173.3 shape=0.77 dur=24; t=2319.116 snr=159.3 shape=0.77 dur=24; t=2351.602 snr=146.2 shape=0.77 dur=24.

Coherence hint: Mixed but usable morphology signal: median shape 0.69 with all rows ingress_egress_ok=True; inspect whether low-shape events are driving noise. Timing hint: Events span 74.8 d with median gap 2.72 d, but 5 close pairs under 0.25 d suggest clustering/cadence checks are important.

Review focus: Start with the top-SNR events, then verify whether lower-SNR events repeat the same morphology and timing pattern. Artifact caution: Stage O flags every row here as A2_severe with whiteness underflow, so red-noise/autocorrelation risk remains high. 5 close event pairs under 0.25 d raise clustering/systematic concerns.

## Rows Most Worth Opening First

Open these first if review time is limited. This is a triage convenience, not a promotion recommendation.

- `EPIC_211972767` (low-score): very high top depth_snr; higher median shape score; many event rows; close-pair clustering adds artifact risk.
- `EPIC_211955365` (median-score): higher median shape score; repeated broad high-SNR events worth checking.
- `EPIC_211530033` (low-score): very high top depth_snr; higher median shape score; close-pair clustering adds artifact risk.
- `EPIC_212003686` (low-score): very high top depth_snr; repeated broad high-SNR events worth checking.
- `EPIC_211759736` (high-score): very high top depth_snr.

## Sparse Or Missing Artifact Context

All 12 rows have `events_csv` and `epic_dir`, but the listed phase/hitmap-style artifact references are blank or missing. This limits visual confidence and makes the event table the primary review aid.

- `EPIC_211529255`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_212007631`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211805106`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211759736`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211530033`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_212003686`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211972767`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211490307`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211955365`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211537297`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211525754`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.
- `EPIC_211421186`: Missing/blank artifact references: best_hits_csv, best_misses_csv, best_uncovered_csv, best_hitmap_png, best_phase_offset_png.

## Practical Review Guidance

- Start with the event table top-SNR rows, then check whether weaker rows repeat the same morphology and timing.
- Treat many very short events as a cadence/single-point artifact risk until inspected.
- Treat many long high-SNR events as a broad-systematics risk until inspected.
- Do not let high `n_events` or high depth SNR override the Stage O `review_high` risk by itself.
- Use the original Stage R sheet for final reviewer annotations; do not infer final outcomes from this assist pack alone.

## Immediate Outcome

The review-assist pack is ready to support manual review. It is descriptive only and makes no final reviewer decisions.
