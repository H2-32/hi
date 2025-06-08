import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from deprnet_model import build_deprnet
from config import WINDOW_X_PATH, WINDOW_Y_PATH

X = np.load(WINDOW_X_PATH)
y = np.load(WINDOW_Y_PATH)

model = build_deprnet()
model.load_weights('deprnet_fold1.h5')  # یا هر fold دلخواه

target_layer = [l.name for l in model.layers if 'max_pooling1d' in l.name][4]
intermediate_layer_model = tf.keras.Model(inputs=model.input,
                                outputs=model.get_layer(target_layer).output)
m5_activations = intermediate_layer_model.predict(X)

# فرض: هر window متعلق به یک subject است (یا بخش‌بندی مناسب انجام شود)
n_subjects = len(np.unique(y))
windows_per_subject = len(y) // n_subjects
response_vectors = []

for subj in range(n_subjects):
    subj_idx = np.where(y == y[subj * windows_per_subject])[0]
    subject_act = m5_activations[subj_idx]
    mean_act = np.mean(subject_act, axis=(0,1))
    response_vectors.append(mean_act)

labels = [int(np.round(np.mean(y[np.where(y == y[s * windows_per_subject])[0]]))) for s in range(n_subjects)]
for i, l in enumerate(labels):
    if l == 0:
        plt.figure()
        sns.heatmap(response_vectors[i][np.newaxis, :], cmap='coolwarm', cbar=True)
        plt.title(f"Subject {i} (Healthy)")
        break
for i, l in enumerate(labels):
    if l == 1:
        plt.figure()
        sns.heatmap(response_vectors[i][np.newaxis, :], cmap='coolwarm', cbar=True)
        plt.title(f"Subject {i} (Depressed)")
        break
plt.show()