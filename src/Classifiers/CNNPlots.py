import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

class CNNPlots:
    def __init__(self):
        pass

    def plot_history(self,history):
        # Accuracy
        plt.figure()
        plt.plot(history.history.get('accuracy', []), label='train')
        plt.plot(history.history.get('val_accuracy', []), label='val')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy'); plt.show()

        # Loss
        plt.figure()
        plt.plot(history.history.get('loss', []), label='train')
        plt.plot(history.history.get('val_loss', []), label='val')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss'); plt.show()


    def plot_roc_pr(self,model, X_test, y_test):
        y_prob = model.predict(X_test).ravel()

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0,1],[0,1],'--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.legend(); plt.show()

        # PR
        p, r, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        plt.figure()
        plt.plot(r, p, label=f'AP = {ap:.3f}')
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall'); plt.legend(); plt.show()

    def safe_plot_roc(self, model, X_test, y_test):
        import matplotlib.pyplot as plt
        if len(np.unique(y_test)) < 2:
            print("ROC undefined: y_test has a single class.")
            # fall back to accuracy or PR for the present class
            y_pred = (model.predict(X_test).ravel() >= 0.5).astype(int)
            acc = (y_pred == y_test).mean()
            print(f"Fallback accuracy: {acc:.3f}")
            return
        y_prob = model.predict(X_test).ravel()
        ap = average_precision_score(y_test, y_prob)
        print(f"Average Precision (AP): {ap:.3f}")
        pos_rate = (y_test==1).mean()
        print(f"AP={ap:.3f} | baseline={pos_rate:.3f}")

        fpr, tpr, thr = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        best = np.argmax(tpr - fpr)          # Youden’s J
        best_thr = thr[best]
        y_pred = (y_prob >= best_thr).astype(int)
        acc = (y_pred == y_test).mean()
        print("Best threshold:", best_thr, "TPR:", tpr[best], "FPR:", fpr[best])
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, digits=3))

        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0,1],[0,1],'--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.show()