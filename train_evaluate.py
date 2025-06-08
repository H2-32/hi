import numpy as np
import tensorflow as tf
from deprnet_model import build_deprnet
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ------------- تنظیمات مسیر و خواندن داده از raw بجای npy -------------

from config import RAW_EEG_PATH, RAW_LABELS_PATH, SELECTED_CHANNELS, FS, WIN_SEC, OVERLAP

# پارامترهای دیتاست MODMA
N_SUBJECTS = 52
N_CHANNELS = 128
N_SAMPLES  = 7680

# خواندن داده خام (raw) و برچسب
eeg_data = np.fromfile(RAW_EEG_PATH, dtype=np.float32).reshape((N_SUBJECTS, N_CHANNELS, N_SAMPLES))
labels = np.fromfile(RAW_LABELS_PATH, dtype=np.int32).reshape((N_SUBJECTS,))

# windowing مشابه preprocess_eeg.py
from sklearn.preprocessing import StandardScaler

def extract_windows(data, label, selected_channels, fs=256, win_len_sec=4, overlap=0.75):
    windows = []
    labels = []
    window_size = int(win_len_sec * fs)
    step = int(window_size * (1 - overlap))
    n_wins = (data.shape[1] - window_size) // step + 1
    for w in range(n_wins):
        seg = data[selected_channels, w*step:w*step+window_size]
        seg = StandardScaler().fit_transform(seg.T).T
        windows.append(seg[..., np.newaxis])
        labels.append(label)
    return windows, labels

all_windows = []
all_labels = []
for subj in range(eeg_data.shape[0]):
    print(f"Extracting windows for subject {subj+1}/{eeg_data.shape[0]}")
    wins, labs = extract_windows(
        eeg_data[subj], labels[subj], SELECTED_CHANNELS, FS, WIN_SEC, OVERLAP)
    all_windows.extend(wins)
    all_labels.extend(labs)

X = np.stack(all_windows)
y = np.array(all_labels)
y_cat = tf.keras.utils.to_categorical(y, num_classes=2)

# --------------------- Cross-validation ------------------------

kf = KFold(n_splits=10, shuffle=True, random_state=42)
metrics_list = []
fold = 1
all_y_true, all_y_pred_prob, all_y_pred_label = [], [], []

for train_index, test_index in kf.split(X, y_cat):
    print(f"Fold {fold}...")
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_cat[train_index], y_cat[test_index]
    val_split = int(0.8 * len(X_train))
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:val_split], indices[val_split:]
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    model = build_deprnet()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    early = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f'deprnet_fold{fold}.h5', save_best_only=True, monitor='val_loss')
    model.fit(X_tr, y_tr, epochs=25, batch_size=64,
              validation_data=(X_val, y_val), callbacks=[early, checkpoint], verbose=2)
    model.load_weights(f'deprnet_fold{fold}.h5')
    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    all_y_true.extend(y_test_labels)
    all_y_pred_prob.extend(y_pred[:,1])
    all_y_pred_label.extend(y_pred_labels)
    report = classification_report(y_test_labels, y_pred_labels, output_dict=True)
    auc = roc_auc_score(y_test, y_pred)
    print(f"Fold {fold}: Accuracy={report['accuracy']:.4f}, AUC={auc:.4f}")
    metrics_list.append({'accuracy': report['accuracy'], 'auc': auc})
    fold += 1

np.save('all_y_true.npy', np.array(all_y_true))
np.save('all_y_pred_prob.npy', np.array(all_y_pred_prob))
np.save('all_y_pred_label.npy', np.array(all_y_pred_label))
np.save('cv_metrics_list.npy', metrics_list)
print("CV Results:")
print("Mean accuracy:", np.mean([m['accuracy'] for m in metrics_list]))
print("Mean AUC:", np.mean([m['auc'] for m in metrics_list]))