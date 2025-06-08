import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, iirnotch
from sklearn.preprocessing import StandardScaler
from config import RAW_EEG_PATH, RAW_LABELS_PATH, SUBJECT_INFO_PATH, SELECTED_CHANNELS, FS, WIN_SEC, OVERLAP, WINDOW_X_PATH, WINDOW_Y_PATH

# تنظیم ابعاد دیتای خام (برای دیتاست MODMA معمولاً 52 نفر، 128 کانال، 7680 نمونه)
N_SUBJECTS = 52
N_CHANNELS = 128
N_SAMPLES  = 7680

# خواندن داده EEG از فایل raw
eeg_data = np.fromfile(RAW_EEG_PATH, dtype=np.float32).reshape((N_SUBJECTS, N_CHANNELS, N_SAMPLES))
labels = np.fromfile(RAW_LABELS_PATH, dtype=np.int32).reshape((N_SUBJECTS,))

# ادامه کد هم مثل قبل!
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass(data, lowcut=0.1, highcut=100, fs=256, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def apply_notch(data, notch_freq=50, fs=256, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return lfilter(b, a, data)

def preprocess_subject(raw, fs):
    for ch in range(raw.shape[0]):
        raw[ch] = apply_bandpass(raw[ch], 0.1, 100, fs)
        raw[ch] = apply_notch(raw[ch], 50, fs)
    return raw

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
    print(f"Preprocessing subject {subj+1}/{eeg_data.shape[0]}")
    d = preprocess_subject(eeg_data[subj], FS)
    wins, labs = extract_windows(d, labels[subj], SELECTED_CHANNELS, FS, WIN_SEC, OVERLAP)
    all_windows.extend(wins)
    all_labels.extend(labs)

X = np.stack(all_windows)
y = np.array(all_labels)
np.save(WINDOW_X_PATH, X)
np.save(WINDOW_Y_PATH, y)
print(f"Done! Saved: {WINDOW_X_PATH}, {WINDOW_Y_PATH}")