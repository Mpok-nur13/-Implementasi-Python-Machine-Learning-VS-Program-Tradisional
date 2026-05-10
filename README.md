# 🤖 Pemrograman Tradisional vs Machine Learning dengan Keras

Notebook ini mendemonstrasikan perbedaan mendasar antara **pemrograman tradisional** dan **pendekatan machine learning** dalam menyelesaikan masalah yang sama — memprediksi output dari sebuah fungsi matematis.

---

## 📌 Deskripsi

Kedua pendekatan digunakan untuk menyelesaikan persamaan berikut:

```
y = 3x + 4
```

- **Pendekatan tradisional**: aturan ditulis langsung oleh programmer.
- **Pendekatan machine learning**: model belajar sendiri menemukan aturan dari data.

---

## 📁 Struktur Notebook

### Cell 1 — Program Tradisional

```python
def rule(data):
    return data * 3 + 4

if __name__ == "__main__":
    data = 5
    answer = rule(data)
    print(answer)  # Output: 19
```

Programmer secara eksplisit mendefinisikan rumus `y = 3x + 4` sebagai sebuah fungsi. Ketika dipanggil dengan input `5`, program langsung menghitung dan mengembalikan `19`.

**Cara kerja**: Input → Aturan (ditulis manual) → Output

---

### Cell 2 — Machine Learning dengan Keras

```python
import keras
import numpy as np

# Membuat model neural network sederhana
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Kompilasi model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Data latih
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([1.0, 4.0, 7.0, 10.0, 13.0, 16.0], dtype=float)

# Melatih model selama 500 epoch
model.fit(xs, ys, epochs=500)

# Prediksi untuk input 5
answer = model.predict(np.array([5.0]))
print(answer)  # Output: ~[[19.0]]
```

Model **tidak diberitahu** rumusnya. Ia hanya diberi pasangan data input (`xs`) dan output (`ys`), lalu belajar sendiri menemukan pola hubungan di antara keduanya selama 500 putaran pelatihan (epoch).

**Cara kerja**: Input + Data Latih → Model Belajar → Output

---

## 🔍 Penjelasan Komponen ML

| Komponen | Penjelasan |
|---|---|
| `keras.Sequential` | Model neural network dengan lapisan-lapisan yang tersusun berurutan |
| `Dense(units=1)` | Satu neuron output — cukup untuk memetakan satu nilai ke satu nilai |
| `optimizer='sgd'` | Stochastic Gradient Descent — algoritma untuk mengupdate bobot model |
| `loss='mean_squared_error'` | Fungsi loss yang mengukur seberapa jauh prediksi dari nilai sesungguhnya |
| `epochs=500` | Model dilatih sebanyak 500 kali melewati seluruh data latih |

---

## 📊 Perbandingan Dua Pendekatan

| Aspek | Tradisional | Machine Learning |
|---|---|---|
| Aturan | Ditulis manual oleh programmer | Dipelajari otomatis dari data |
| Fleksibilitas | Kaku, harus diubah jika pola berubah | Adaptif terhadap data baru |
| Kebutuhan data | Tidak perlu data latih | Membutuhkan data latih |
| Akurasi output | Tepat (19) | Mendekati (±19.01) |
| Cocok untuk | Masalah dengan aturan yang jelas | Masalah dengan pola kompleks/tidak diketahui |

---

## ⚙️ Requirements

```
tensorflow / keras
numpy
```

Install dengan:

```bash
pip install tensorflow numpy
```

---

## 🚀 Cara Menjalankan

1. Buka notebook di [Google Colab](https://colab.research.google.com/) atau Jupyter.
2. Jalankan Cell 1 untuk melihat hasil pemrograman tradisional.
3. Jalankan Cell 2 untuk melatih model dan melihat prediksi ML.

> **Catatan**: Cell 2 menggunakan GPU (T4) di Google Colab untuk mempercepat proses pelatihan.

---

## 💡 Insight

Meskipun kedua pendekatan menghasilkan jawaban yang sama (~19 untuk input 5), machine learning melakukannya **tanpa mengetahui rumusnya terlebih dahulu**. Ini adalah inti dari konsep machine learning: membiarkan model menemukan pola dari data, bukan memprogram aturannya secara eksplisit.
