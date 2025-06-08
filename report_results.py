import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

metrics_list = np.load('cv_metrics_list.npy', allow_pickle=True)
all_y_true = np.load('all_y_true.npy')
all_y_pred_prob = np.load('all_y_pred_prob.npy')
all_y_pred_label = np.load('all_y_pred_label.npy')

print("==== Cross-validation results ====")
print(f"Mean accuracy: {np.mean([m['accuracy'] for m in metrics_list]):.4f}")
print(f"Mean AUC: {np.mean([m['auc'] for m in metrics_list]):.4f}")

print("Confusion matrix (pooled):")
print(confusion_matrix(all_y_true, all_y_pred_label))

# Plot ROC
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_prob)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve (DeprNet)')
plt.legend(loc="lower right")
plt.show()