# EEG Depression Classification with DeprNet (MODMA Dataset)

این پروژه، یک چارچوب جامع برای پیش‌پردازش، تقسیم‌بندی، آموزش، ارزیابی و تحلیل داده‌های EEG جهت طبقه‌بندی اختلال افسردگی با استفاده از شبکه DeprNet و دیتاست MODMA است.

---

## نیازمندی‌ها

- Python 3.7+
- کتابخانه‌های مورد نیاز:
  - numpy
  - scipy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - tensorflow 2.x

برای نصب وابستگی‌ها:
```bash
pip install numpy scipy pandas scikit-learn matplotlib seaborn tensorflow
```

---

## آماده‌سازی پایگاه داده (دیتابیس)

۱. **دریافت دیتاست MODMA:**
   - دیتاست MODMA را از [لینک رسمی MODMA](https://doi.org/10.6084/m9.figshare.16695853) دریافت کنید.
   - فایل‌های زیر باید موجود باشند:
     - **داده EEG خام:**  
       `eeg_data_raw.npy` (ابعاد: تعداد افراد × تعداد کانال × نمونه)
     - **برچسب هر فرد:**  
       `labels_raw.npy` (۰=سالم، ۱=افسرده)
     - **(اختیاری) اطلاعات تکمیلی افراد:**  
       `subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx`

۲. **ساختار فایل‌ها:**
   - فایل‌ها را در ریشه پروژه قرار دهید یا مسیر آن‌ها را مطابق با `config.py` تنظیم کنید.

---

## اجرای پروژه (گام به گام)

### ۱. پیش‌پردازش داده‌ها و تقسیم به window

این مرحله داده خام را فیلتر، نرمالایز و به windowهای ۴ثانیه با همپوشانی ۷۵٪ تقسیم می‌کند.

```bash
python preprocess_eeg.py
```
خروجی:  
- `X_modma_deprnet.npy`  
- `y_modma_deprnet.npy`

---

### ۲. آموزش و اعتبارسنجی مدل DeprNet

این مرحله با cross-validation ده‌برابرانه مدل DeprNet را آموزش داده و وزن بهترین مدل هر fold را ذخیره می‌کند.

```bash
python train_evaluate.py
```
خروجی:
- وزن مدل برای هر fold (مثلاً: `deprnet_fold1.h5`)
- فایل‌های نتایج:  
  - `cv_metrics_list.npy`  
  - `all_y_true.npy`  
  - `all_y_pred_prob.npy`  
  - `all_y_pred_label.npy`

---

### ۳. تحلیل لایه‌های شبکه (featuremap analysis)

تحلیل پاسخ آخرین maxpooling لایه شبکه و نمایش heatmap برای نمونه‌های سالم و افسرده.

```bash
python featuremap_analysis.py
```
خروجی:  
- نمودارهای heatmap پاسخ کانال‌ها

---

### ۴. گزارش و ترسیم نتایج نهایی

نمایش دقت، AUC، ماتریس آشفتگی و نمودار ROC.

```bash
python report_results.py
```
خروجی:  
- میانگین دقت و AUC کل folds
- ماتریس confusion
- نمودار ROC

---

## نکته‌ها و شخصی‌سازی

- اگر ساختار یا نام فایل‌ها متفاوت است، پارامترهای مسیر را در فایل `config.py` تنظیم کنید.
- کدها برای توسعه آسان (مثلاً تحلیل ablation، استخراج ویژگی ERP و ...) آماده هستند.
- در صورت وجود سوال یا نیاز به توسعه بیشتر با ما تماس بگیرید.

---

## ساختار پوشه پروژه

```
├── config.py
├── preprocess_eeg.py
├── deprnet_model.py
├── train_evaluate.py
├── featuremap_analysis.py
├── report_results.py
├── eeg_data_raw.npy
├── labels_raw.npy
└── subjects_information_EEG_128channels_ERP_lanzhou_2015.xlsx (اختیاری)
```

---

## اعتبارسنجی و استناد

- لطفاً در مقالات و گزارش‌های خود به دیتاست MODMA و مقاله DeprNet استناد کنید.
- ساختار پیش‌فرض کد مطابق با استانداردهای پژوهشی و قابل استفاده جهت انتشار مقاله است.

---