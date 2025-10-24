
Training On TESS Data .
-----------------------

Still more to work on AUC.

Target → light curve: For each star (prefer TIC id), we downloaded its TESS light curve and cleaned it (NaNs out, normalize).

Segmenting: Chopped (e.g., 200 points) and z-scored each window. That gives the CNN uniform, comparable snippets.

Labels: 
Transit → class 1 (transit-like), anything else → class 0 (non-transit).

Group-aware split: Kept all segments from the same star on one side (train or test) so the model can’t “cheat” by seeing nearly identical windows from the same source. 
Model (1D-CNN): A small stack of 1D convolutions + pooling, then global average pooling and a sigmoid output. Convs learn the local shape of a transit dip; pooling/GAP compress to a decision.

Training & eval: Accuracy/ROC/PR on held-out stars; optional per-star scoring by aggregating segment predictions.

-------- Took a lot to improvise splits.