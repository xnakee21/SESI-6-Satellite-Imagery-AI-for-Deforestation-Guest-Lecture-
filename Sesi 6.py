import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Simulasi Ekstraksi Piksel Citra Satelit Multi-Spektral
# Fitur: [Band 4 (Red), Band 3 (Green), Band 8 (NIR), Nilai_NDVI]
np.random.seed(42)
n_pixels = 300

# Hutan Utuh: NIR Tinggi, NDVI Sangat Tinggi
hutan = np.hstack([
    np.random.normal(0.05, 0.01, (100, 1)),
    np.random.normal(0.08, 0.01, (100, 1)),
    np.random.normal(0.50, 0.03, (100, 1)),
    np.random.normal(0.80, 0.04, (100, 1))
])

# Bekas Kebakaran: Red Tinggi, NIR Rendah, NDVI Rendah/Negatif
kebakaran = np.hstack([
    np.random.normal(0.30, 0.02, (100, 1)),
    np.random.normal(0.25, 0.02, (100, 1)),
    np.random.normal(0.12, 0.02, (100, 1)),
    np.random.normal(0.05, 0.03, (100, 1))
])

# Perkebunan Sawit: NIR Sedang, NDVI Sedang-Tinggi (Pola Teratur)
sawit = np.hstack([
    np.random.normal(0.12, 0.01, (100, 1)),
    np.random.normal(0.15, 0.01, (100, 1)),
    np.random.normal(0.35, 0.02, (100, 1)),
    np.random.normal(0.55, 0.03, (100, 1))
])

X_pixels = np.vstack([hutan, kebakaran, sawit])
y_pemicu = np.array([0]*100 + [1]*100 + [2]*100) # 0: Hutan Utuh, 1: Kebakaran, 2: Sawit

# 2. Pelatihan Model Random Forest Spasial
model_deforest = RandomForestClassifier(n_estimators=100, random_state=42)
model_deforest.fit(X_pixels, y_pemicu)

# 3. Pengujian pada Piksel Peringatan Deforestasi Baru (Deforestation Alert)
# Skenario: Deteksi Citra Baru [Red, Green, NIR, NDVI]
piksel_alert_1 = np.array([[0.29, 0.24, 0.11, 0.04]]) # Skenario Kebakaran
piksel_alert_2 = np.array([[0.13, 0.14, 0.36, 0.54]]) # Skenario Sawit

pred_1 = model_deforest.predict(piksel_alert_1)
pred_2 = model_deforest.predict(piksel_alert_2)

map_kategori = {0: 'Hutan Utuh (Tidak Ada Deforestasi)', 1: 'Pemicu: Kebakaran Hutan', 2: 'Pemicu: Ekspansi Perkebunan Sawit'}

print("=== HASIL KLASIFIKASI PEMICU DEFORESTASI DARI CITRA SATELIT ===")
print(f"Peringatan Spasial 1 -> Terdeteksi sebagai: {map_kategori[pred_1[0]]}")
print(f"Peringatan Spasial 2 -> Terdeteksi sebagai: {map_kategori[pred_2[0]]}")

# Tingkat Kepentingan Fitur Spektral
print("\n=== KONTRIBUSI FITUR SPEKTRAL TERHADAP PREDIKSI (FEATURE IMPORTANCE) ===")
fitur_names = ['Red Band', 'Green Band', 'NIR Band', 'Indeks NDVI']
for nama, imp in zip(fitur_names, model_deforest.feature_importances_):
    print(f"Atribut {nama:12s}: {imp*100:.2f}%")
