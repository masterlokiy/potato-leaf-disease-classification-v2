import streamlit as st
import torch
import timm
import numpy as np
from PIL import Image
import time
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os

# =====================================================================================
# KONFIGURASI HALAMAN & GAYA
# =====================================================================================
st.set_page_config(
    page_title="SpudScan: Diagnosis Cerdas Daun Kentang",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================================
# DATABASE INFORMASI PENYAKIT (KUNCI DISESUAIKAN DENGAN OUTPUT MODEL)
# =====================================================================================
DISEASE_INFO = {
    "Healthy": {
        "description": "Tanaman dalam kondisi prima tanpa tanda-tanda penyakit yang terlihat.",
        "symptoms": "Warna daun hijau segar dan merata, bentuk daun normal dan utuh, tidak ada bercak atau lubang.",
        "treatment": "Tidak ada tindakan pengobatan yang diperlukan. Pertahankan kondisi optimal.",
        "prevention": "Lanjutkan praktik pertanian yang baik: pemupukan seimbang, irigasi cukup, dan pemantauan rutin.",
        "color": "#4CAF50",  # Hijau
        "icon": "✅"
    },
    "Virus": {
        "description": "Infeksi oleh partikel virus yang mengganggu metabolisme dan pertumbuhan normal tanaman.",
        "symptoms": "Daun menguning dengan pola mosaik (belang-belang), daun keriput atau menggulung, pertumbuhan tanaman kerdil.",
        "treatment": "Tidak ada obat untuk virus. Cabut dan musnahkan tanaman terinfeksi untuk mencegah penyebaran. Kendalikan serangga vektor.",
        "prevention": "Gunakan bibit bersertifikat bebas virus, sanitasi alat pertanian, dan kendalikan populasi kutu daun (aphid).",
        "color": "#9C27B0",  # Ungu
        "icon": "🦠"
    },
    "Phytophthora": {
        "description": "Dikenal sebagai busuk daun (late blight), disebabkan oleh oomycete Phytophthora infestans. Sangat destruktif dalam kondisi lembab dan sejuk.",
        "symptoms": "Bercak basah berwarna hijau keabu-abuan pada tepi daun, yang cepat membesar menjadi coklat kehitaman. Sisi bawah daun seringkali memiliki lapisan jamur putih.",
        "treatment": "Aplikasi fungisida sistemik dan kontak secara berkala. Pemangkasan bagian yang terinfeksi parah.",
        "prevention": "Jaga sirkulasi udara yang baik, hindari penyiraman dari atas daun, lakukan rotasi tanaman, gunakan varietas tahan.",
        "color": "#795548",  # Coklat
        "icon": "🍂"
    },
    "Nematode": {
        "description": "Serangan oleh cacing mikroskopis (nematoda) pada akar yang merusak sistem perakaran dan penyerapan nutrisi.",
        "symptoms": "Tanaman tampak layu pada siang hari meskipun tanah cukup basah, daun menguning, pertumbuhan kerdil, dan terdapat bintil (sista) pada akar.",
        "treatment": "Penggunaan nematisida, solarisasi tanah sebelum tanam, atau menanam tanaman perangkap (trap crops).",
        "prevention": "Gunakan bibit bebas nematoda, lakukan rotasi tanaman dengan tanaman non-inang seperti jagung, dan tingkatkan bahan organik tanah.",
        "color": "#FFC107",  # Kuning Amber
        "icon": "🪱"
    },
    "Fungi": {
        "description": "Infeksi oleh jamur seperti Alternaria solani (early blight). Menyerang daun, batang, dan umbi.",
        "symptoms": "Bercak gelap, kering, dengan pola lingkaran konsentris seperti target tembak. Biasanya dimulai dari daun bagian bawah.",
        "treatment": "Aplikasi fungisida yang sesuai. Buang dan musnahkan daun yang terinfeksi untuk mengurangi sumber spora.",
        "prevention": "Pastikan jarak tanam yang cukup untuk sirkulasi udara, hindari kelembaban berlebih pada daun, dan lakukan rotasi tanaman.",
        "color": "#FF9800",  # Oranye
        "icon": "🍄"
    },
    "Bacteria": {
        "description": "Infeksi oleh bakteri patogen seperti Ralstonia solanacearum (layu bakteri) atau Pectobacterium carotovorum (busuk lunak).",
        "symptoms": "Layu mendadak pada seluruh tanaman tanpa daun menguning terlebih dahulu. Jika batang dipotong, akan keluar lendir putih susu.",
        "treatment": "Sulit dikendalikan. Cabut dan hancurkan tanaman yang terinfeksi. Gunakan bakterisida berbasis tembaga sebagai pencegahan.",
        "prevention": "Gunakan bibit sehat, sanitasi alat pertanian, perbaiki drainase tanah, dan lakukan rotasi tanaman.",
        "color": "#2196F3",  # Biru
        "icon": "🧫"
    },
    "Pest": {
        "description": "Kerusakan fisik pada daun yang disebabkan oleh serangan serangga pengunyah atau pengisap.",
        "symptoms": "Daun berlubang, tepi daun rusak, terdapat jejak gigitan, atau adanya serangga, telur, atau kotorannya pada daun.",
        "treatment": "Gunakan insektisida yang sesuai (organik atau kimia), atau metode pengendalian hayati dengan melepaskan predator alami.",
        "prevention": "Pemantauan rutin, pemasangan perangkap serangga, dan menjaga kebersihan area tanam dari gulma.",
        "color": "#F44336",  # Merah
        "icon": "🐛"
    }
}

# Nama kelas sesuai urutan pelatihan model ViT
CLASS_NAMES_ID = ['Bakteri', 'Jamur', 'Healthy', 'Nematode', 'Pest', 'Phytophthora', 'Virus']
# Mapping dari Bahasa Indonesia ke kunci Bahasa Inggris di DISEASE_INFO
CLASS_MAP_EN = {'Bakteri': 'Bacteria', 'Jamur': 'Fungi', **{n:n for n in CLASS_NAMES_ID if n not in ['Bakteri', 'Jamur']}}

# =====================================================================================
# CSS KUSTOM
# =====================================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4f0fb 100%);
    }
    
    .stApp {
        background-color: transparent;
    }
    
    .header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        background: linear-gradient(90deg, #ff7e5f 0%, #feb47b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: fadeIn 1s ease-in-out;
    }

    .header p {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
    }
    
    .result-box {
        padding: 1.5rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
        border-top: 5px solid var(--disease-color, #4e73df);
        height: 100%;
    }
    
    .confidence-bar-container {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
    }

    .confidence-bar {
        height: 10px;
        width: 100%;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        background-color: var(--disease-color, #4e73df);
        width: var(--confidence, 0%);
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        padding: 10px 15px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }

    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }

    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideIn {
        from { transform: translateY(20px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================================
# LOGIKA MODEL (ViT)
# =====================================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 7

@st.cache_resource
def load_vit_model():
    """Memuat model Vision Transformer yang telah dilatih."""
    try:
        model = timm.create_model('vit_base_patch16_224_in21k', pretrained=False, num_classes=NUM_CLASSES, drop_rate=0.3)
        model.head = nn.Sequential(nn.LayerNorm(model.head.in_features), nn.Dropout(0.3), nn.Linear(model.head.in_features, NUM_CLASSES))
        model.load_state_dict(torch.load('model\MODEL93VIT.pth', map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("File model 'MODEL93VIT.pth' tidak ditemukan. Pastikan file berada di direktori yang sama dengan skrip.")
        return None
    except Exception as e:
        st.error(f"Gagal memuat model ViT: {e}")
        return None

# Transformasi gambar agar sesuai dengan input model ViT
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_with_vit(image_pil, model):
    """Melakukan prediksi pada satu gambar menggunakan model ViT."""
    img_tensor = TRANSFORM(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
    
    pred_class_id = CLASS_NAMES_ID[pred_idx.item()]
    confidence = conf.item()
    all_probs = probs.cpu().numpy().flatten()
    
    confidence_scores = {CLASS_NAMES_ID[i]: float(score) for i, score in enumerate(all_probs)}
    return pred_class_id, confidence, confidence_scores

# =====================================================================================
# FUNGSI TAMPILAN
# =====================================================================================
def show_detailed_info(disease_name_id):
    """Menampilkan informasi detail penyakit dalam format tab."""
    disease_key = CLASS_MAP_EN.get(disease_name_id, "Healthy")
    info = DISEASE_INFO.get(disease_key)
    if not info:
        st.warning("Informasi detail untuk penyakit ini tidak tersedia.")
        return
        
    st.markdown(f"### {info['icon']} Informasi Detail: {disease_name_id}")
    tab1, tab2, tab3, tab4 = st.tabs(["Deskripsi", "Gejala Umum", "Cara Perawatan", "Tips Pencegahan"])
    
    with tab1:
        st.write(info["description"])
    with tab2:
        st.write(info["symptoms"])
    with tab3:
        st.write(info["treatment"])
    with tab4:
        st.write(info["prevention"])

def render_confidence_bar(confidence, color):
    """Render batang persentase keyakinan dengan warna khusus."""
    pct = int(confidence * 100)
    bar_html = f"""
    <div style="background:#e0e0e0; border-radius:10px; overflow:hidden; width:100%; height:14px;">
      <div style="width:{pct}%; background:{color}; height:14px;"></div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

def main_page(model):
    st.markdown('<div class="header"><h1>🌿 SpudScan</h1><p>Diagnosis Penyakit Daun Kentang dengan Vision Transformer</p></div>', unsafe_allow_html=True)
    st.write("---")
    
    uploaded_file = st.file_uploader("Unggah Gambar Daun Kentang", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert('RGB')
        except Exception as e:
            st.error(f"Gagal membuka gambar: {e}")
            return
        
        # Membuat layout dengan 2 kolom
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            with st.spinner("Mendeteksi penyakit..."):
                time.sleep(0.1)  # simulasi delay loading
                pred_class, confidence, all_confidences = predict_with_vit(image, model)
            
            disease_key = CLASS_MAP_EN.get(pred_class, "Healthy")
            color = DISEASE_INFO[disease_key]["color"]

            st.markdown(f'<div class="result-box" style="border-top-color:{color};">', unsafe_allow_html=True)
            st.markdown(f"## {DISEASE_INFO[disease_key]['icon']} Prediksi: **{pred_class}**")
            st.markdown(f"### Tingkat Keyakinan: {confidence*100:.2f}%")
            render_confidence_bar(confidence, color)
            show_detailed_info(pred_class)
            st.markdown("</div>", unsafe_allow_html=True)

    else:
        st.info("Silakan unggah gambar daun kentang untuk memulai deteksi penyakit.")

# =====================================================================================
# MAIN PROGRAM
# =====================================================================================
def main():
    with st.sidebar:
        try:
            st.markdown(
                """
                <h1 style='text-align: center; margin-bottom: 0;'>SpudScan</h1>
                """,
                unsafe_allow_html=True
            )
            st.image("assets\2.png", use_container_width=True)
        except:
            st.markdown(
                """
                <h1 style='text-align: center; margin-bottom: 0;'>SpudScan</h1>
                """,
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown(
            """
            **Model:** ViT pretrained on ImageNet-21k  
            **Akurasi:** 93% 
            **Pengembang:** [masterlokiy](https://github.com/masterlokiy)
            """
        )
        st.markdown("---")
        st.info("© 2025 - Dibangun dengan Streamlit & PyTorch.")

    model = load_vit_model()
    if model is None:
        st.stop()

    main_page(model)

if __name__ == "__main__":
    main()