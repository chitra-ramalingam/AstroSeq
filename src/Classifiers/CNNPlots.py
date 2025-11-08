import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    classification_report, confusion_matrix,
    accuracy_score, balanced_accuracy_score, f1_score
)

class CNNPlots:
    """Utility plots + thresholded evaluation for binary classifiers."""

    # -------- history --------
    def plot_history(self, history):
        h = getattr(history, "history", {})
        def _maybe_plot(keys, title, ylabel):
            vals = [k for k in keys if k in h]
            if not vals: return
            plt.figure()
            for k in vals:
                plt.plot(h[k], label=k)
            plt.xlabel('Epoch'); plt.ylabel(ylabel)
            plt.title(title); plt.legend(); plt.show()

        _maybe_plot(['accuracy','val_accuracy'], 'Accuracy', 'Accuracy')
        _maybe_plot(['loss','val_loss'], 'Loss', 'Loss')
        _maybe_plot(['roc_auc','val_roc_auc'], 'ROC-AUC', 'AUC')
        _maybe_plot(['pr_auc','val_pr_auc'], 'PR-AUC', 'AP')

    # -------- ROC & PR --------
    def plot_roc_pr(self, model, X, y, set_name="TEST"):
        y = np.asarray(y).astype(int)
        if np.unique(y).size < 2:
            print(f"[{set_name}] ROC/PR undefined: only one class in y.")
            return

        y_prob = model.predict(X, verbose=0).ravel().astype(float)
        if not np.isfinite(y_prob).all():
            raise ValueError("NaN/Inf detected in predicted probabilities.")

        # ROC
        fpr, tpr, _ = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0,1], [0,1], '--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC — {set_name}')
        plt.legend(); plt.show()

        # PR
        p, r, _ = precision_recall_curve(y, y_prob)
        ap = average_precision_score(y, y_prob)
        baseline = y.mean()
        plt.figure()
        plt.plot(r, p, label=f'AP = {ap:.3f}')
        plt.hlines(baseline, 0, 1, linestyles='--', label=f'Baseline={baseline:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision–Recall — {set_name}')
        plt.legend(); plt.show()

        print(f"[{set_name}] ROC-AUC={roc_auc:.3f} | PR-AUC={ap:.3f} | PosRate={baseline:.3f}")

    # -------- robust ROC with best threshold & report --------
    def safe_plot_roc(self, model, X, y, set_name="TEST"):
        y = np.asarray(y).astype(int)
        if np.unique(y).size < 2:
            print(f"[{set_name}] ROC undefined: y has a single class.")
            y_pred = (model.predict(X, verbose=0).ravel() >= 0.5).astype(int)
            acc = accuracy_score(y, y_pred)
            print(f"[{set_name}] Fallback accuracy@0.5: {acc:.3f}")
            return

        y_prob = model.predict(X, verbose=0).ravel().astype(float)
        ap = average_precision_score(y, y_prob)
        pos_rate = y.mean()
        print(f"[{set_name}] AP={ap:.3f} | baseline={pos_rate:.3f}")

        fpr, tpr, thr = roc_curve(y, y_prob)
        roc_auc = auc(fpr, tpr)

        # Youden’s J (guard if thr empty)
        if thr.size:
            j = tpr - fpr
            best_idx = int(np.argmax(j))
            best_thr = float(thr[best_idx])
            y_pred = (y_prob >= best_thr).astype(int)
            print(f"[{set_name}] Best thr (Youden J): {best_thr:.4f}  TPR={tpr[best_idx]:.3f}  FPR={fpr[best_idx]:.3f}")
            print(confusion_matrix(y, y_pred))
            print(classification_report(y, y_pred, digits=3))
        else:
            print(f"[{set_name}] Could not derive thresholds from ROC.")

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC — {set_name}")
        plt.legend(); plt.show()

    # -------- choose threshold on VAL + evaluate on TEST --------
    def choose_threshold_from_val(self, y_val, p_val, mode="balanced_accuracy"):
        y_val = np.asarray(y_val).astype(int)
        p_val = np.asarray(p_val).astype(float)
        pr, rc, th = precision_recall_curve(y_val, p_val)  # th aligns with pr[1:], rc[1:]
        if th.size == 0: 
            return 0.5
        if mode == "f1":
            f1 = (2*pr[1:]*rc[1:])/(pr[1:]+rc[1:]+1e-9)
            return float(th[np.argmax(f1)])
        if mode == "balanced_accuracy":
            scores = [balanced_accuracy_score(y_val, (p_val>=t).astype(int)) for t in th]
            return float(th[int(np.argmax(scores))])
        # default: accuracy
        accs = [accuracy_score(y_val, (p_val>=t).astype(int)) for t in th]
        return float(th[int(np.argmax(accs))])

    def evaluate_with_threshold(self, model, X_val, y_val, X_test, y_test, mode="balanced_accuracy"):
        p_val  = model.predict(X_val,  verbose=0).ravel().astype(float)
        p_test = model.predict(X_test, verbose=0).ravel().astype(float)
        thr = self.choose_threshold_from_val(y_val, p_val, mode=mode)
        print(f"Chosen threshold ({mode} on VAL): {thr:.4f}")

        def _report(y, p, name):
            y = np.asarray(y).astype(int)
            yhat = (p >= thr).astype(int)
            print(f"\n[{name}] pos_rate true/pred: {y.mean():.3f} / {yhat.mean():.3f}")
            print("accuracy:", accuracy_score(y, yhat))
            print("balanced_accuracy:", balanced_accuracy_score(y, yhat))
            print("f1:", f1_score(y, yhat, zero_division=0))
            print(confusion_matrix(y, yhat))
            print(classification_report(y, yhat, digits=3))

        _report(y_val,  p_val,  "VAL")
        _report(y_test, p_test, "TEST")
        return thr
