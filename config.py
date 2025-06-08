# config.py
# مسیرها و تنظیمات اصلی پروژه را اینجا قرار دهید.

# مسیر فایل‌های خام (raw) که از دیتاست MODMA استخراج شده‌اند
RAW_EEG_PATH = 'drive-download-20250607T190931Z-1-003/eeg_data_raw.raw'
RAW_LABELS_PATH = 'drive-download-20250607T190931Z-1-003/labels_raw.raw'
SUBJECT_INFO_PATH = 'drive-download-20250607T190931Z-1-003/subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx'  # (اختیاری)

# پارامترهای پردازش
SELECTED_CHANNELS = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]  # 19 کانال از 128 کانال
FS = 256
WIN_SEC = 4
OVERLAP = 0.75

# اگر نمی‌خواهی هیچ فایل میانی ذخیره کنی، این دو خط را حذف کن
# WINDOW_X_PATH = 'X_modma_deprnet.npy'
# WINDOW_Y_PATH = 'y_modma_deprnet.npy'