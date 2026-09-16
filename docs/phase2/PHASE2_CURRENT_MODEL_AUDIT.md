# Phase 2 Current Model Audit

## Selected frozen morphology encoder

Freeze `models/k2_nocrop_flux_seed46_split303.best.keras`. This is the active model named by `freezes/k2_flux_model_official_policy_note.txt:8`, `freezes/stage_f_closed_manifest.txt:45`, and the scoring constant in `scripts/backfill_master_catalog_cnn_and_manual_queue.py:14`. Historical root and mission models are competing artifacts, but repository policy does not designate them as current.

Direct load inspection (TensorFlow, `compile=False`) gives input `(None, 512, 1)` and output `(None, 1)`. The file SHA-256 is `547e278e436d91165ccd4f18cee2562d4a9befbbf8a2de7bb06357cda88b4443`. Layers are Conv1D(64,k=11) → LayerNorm → max pool → Conv1D(128,k=7) → LayerNorm → max pool → Conv1D(128,k=5) → LayerNorm → max pool → Conv1D(128,k=3) → global average pooling → dropout(0.2) → Dense(1,sigmoid). The final layer is `dense_2`; the penultimate layer is `dropout_2`, dimension 128. Prefer the deterministic pre-dropout `global_average_pooling1d_2` at inference (also 128-D); with `training=False`, dropout output is identical but the pooling layer is semantically cleaner.

## Input, preprocessing, and target

`K2_training_main.py:17-18,45-78` fixes split seed 303, train seed 46, Campaign 5, 512-sample windows, stride 256, default preprocessing and synthetic injection. `src/Classifiers/K2/K2_Dataset_builder.py:15-47,170-222` flattens (window 401, polynomial order 2), converts to relative flux, injects box transits, robust-centres/MAD-scales, clips at 10 sigma, and fills non-finite values with zero. `src/Classifiers/K2/K2_trainer.py:42-50` selects flux channel 0 and does not crop. It is therefore an unfurled, fixed-length time-series segment model—not a P-fold, local-transit, or global-fold model.

The target is segment `meta['label']`, with positives created from injected box transits on about half of Campaign-5 stars (`K2_Dataset_builder.py:52-65,128-170`); the trainer balances positive and negative streams (`K2_trainer.py:69-94`). Binary cross-entropy and a sigmoid output are declared by `src/Classifiers/Builders/BuilderHelper.py`. Output is `flux_p_science_like` historically and `transit_morphology_score` in the master-catalog backfill; policy says it is dip/transit morphology only, never auto-promotion (`freezes/k2_flux_model_official_policy_note.txt:12-27`; `backfill_master_catalog_cnn_and_manual_queue.py:21-25`).

## Weight-preserving embedding export

```python
import numpy as np
import tensorflow as tf

path = "models/k2_nocrop_flux_seed46_split303.best.keras"
model = tf.keras.models.load_model(path, compile=False)
model.trainable = False
encoder = tf.keras.Model(model.input, model.get_layer("global_average_pooling1d_2").output)
x = np.asarray(x_512_flux[:, :, :1], dtype=np.float32)
p = model.predict(x, batch_size=256, verbose=0).reshape(-1)
z = encoder.predict(x, batch_size=256, verbose=0)  # (N, 128)
```

This creates a read-only view of existing weights. Aggregate multiple segment embeddings per EPIC explicitly (for example score-weighted mean plus max-score segment), retaining segment count and aggregation policy.

## Competing models

The repository contains numerous `.keras` files (root mission baselines, hard-negative variants, `best.keras`, `best_ref.keras`, and `models/*`). `K2_training_main.py` and three freeze/policy files resolve the ambiguity in favour of the exact `.best.keras` path above. Do not substitute the similarly named non-best checkpoint.
