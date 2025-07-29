# 🥔 SpudScan: Deteksi Cerdas Penyakit Daun Kentang

**SpudScan** adalah aplikasi berbasis web interaktif yang dirancang untuk membantu petani dan peneliti mendiagnosis penyakit pada daun kentang secara otomatis menggunakan model deep learning **Vision Transformer (ViT)**. Aplikasi ini dibangun dengan framework **Streamlit** yang ringan dan mudah digunakan.

---

## 🚀 Fitur Utama

- 🔍 **Deteksi cepat** dari 7 jenis penyakit daun kentang
- 🧠 **Model akurat** berbasis Vision Transformer (ViT)
- 🌱 Antarmuka intuitif berbasis web
- 📤 Upload gambar langsung untuk diagnosis instan
- 📊 Dirancang untuk petani, peneliti, dan komunitas pertanian

---

## 🧪 Kelas Penyakit yang Didukung

| Label Indonesia | Label Inggris (Opsional) |
| --------------- | ------------------------ |
| Bakteri         | Bacteria                 |
| Jamur           | Fungi                    |
| Sehat           | Healthy                  |
| Nematoda        | Nematode                 |
| Hama            | Pest                     |
| Phytophthora    | Phytophthora             |
| Virus           | Virus                    |

---

---

## ⚙️ Cara Menjalankan Aplikasi

1. **Clone repo**

```bash
git clone https://github.com/username/spudscan.git
cd spudscan
```

2. **Install dependensi**

```bash
pip install -r requirements.txt
```

3. **Jalankan Streamlit**

```bash
streamlit run app.py
```

---

## 🧠 Teknologi yang Digunakan

- Python 3.8+
- Streamlit
- PyTorch
- timm (PyTorch image models)
- Torchvision, NumPy, Pillow (PIL)

---

## 📚 Dataset

Dataset terdiri dari 3.076 gambar daun kentang yang dikategorikan ke dalam 7 kelas penyakit.

---

## 📸 Contoh Prediksi

![Contoh Prediksi](assets/sample_leaf.jpg)

---

## 🧑‍💻 Kontributor

- Ilham C.S. Wibowo – [@masterlokiy](https://github.com/masterlokiy)

---

## 📄 Lisensi

MIT License — Silakan digunakan untuk pembelajaran, riset, atau pengembangan pertanian digital lebih lanjut.

---
````
