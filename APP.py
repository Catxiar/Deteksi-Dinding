import streamlit as st
from ultralytics import YOLO # Diaktifkan kembali untuk menggunakan model .pt
import cv2
import numpy as np
import random
import io
import os # Diaktifkan kembali untuk memeriksa file model

# --- ⚙️ KONFIGURASI ---
MODEL_PATH = 'best.pt' # Menggunakan model custom yang terlatih

# Memuat model YOLOv8 hanya sekali (caching)
@st.cache_resource
def load_yolo_model(path):
    """Memuat model YOLO menggunakan cache Streamlit."""
    if not os.path.exists(path):
        # Menggunakan st.error() untuk pesan kegagalan memuat model
        st.error(f"❌ Gagal memuat model. File '{path}' tidak ditemukan.")
        st.info("Pastikan file 'best.pt' (model YOLOv8 terlatih) ada di direktori yang sama.")
        return None
        
    try:
        # Pemuatan model
        model = YOLO(path)
        st.success(f"✅ Model YOLOv8 (best.pt) berhasil dimuat.")
        return model
    except Exception as e:
        # Menangani kegagalan inisialisasi model
        st.error(f"❌ Terjadi kesalahan saat memuat model YOLOv8: {e}")
        return None

def run_yolo_inference(model, image_file, conf_threshold):
    """Menjalankan inferensi deteksi kerusakan (localization) menggunakan model custom."""
    
    # 1. Konversi Streamlit UploadedFile ke format OpenCV (BGR)
    image_bytes = io.BytesIO(image_file.read())
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return None, "Gagal membaca citra. Pastikan format file valid.", 0

    # 2. Jalankan Prediksi YOLO
    with st.spinner(f'⏳ Menjalankan deteksi kerusakan dengan Conf={conf_threshold:.2f}...'):
        try:
            # Jalankan prediksi dengan model best.pt
            results = model.predict(
                source=img_bgr, 
                conf=conf_threshold,
                iou=0.5, # Ambil IOU yang umum
                save=False,
                imgsz=640,
                verbose=False
            )
        except Exception as e:
            return None, f"Gagal menjalankan prediksi YOLO: {e}", 0
            
    if not results:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb, "Tidak ada hasil prediksi kerusakan yang dikembalikan.", 0

    result = results[0]
    
    # 3. Visualisasi dan Konversi
    # result.plot() MENGHASILKAN CITRA DENGAN BOUNDING BOX.
    plotted_image_bgr = result.plot()
    plotted_image_rgb = cv2.cvtColor(plotted_image_bgr, cv2.COLOR_BGR2RGB)
    
    # 4. Kumpulkan ringkasan deteksi
    
    # Pengecekan eksplisit terhadap result.boxes untuk menghindari error NoneType
    detections = len(result.boxes) if (result.boxes is not None and len(result.boxes) > 0) else 0
    
    info = f"**Total Kerusakan Terdeteksi:** **{detections}**\n"
    
    if detections > 0:
        info += "\n**Ringkasan Deteksi:**\n"
        
        # Hitung frekuensi setiap kelas
        class_counts = {}
        # result.boxes dijamin tidak None karena detections > 0
        for cls_id in result.boxes.cls.cpu().numpy():
            # Mengambil nama kelas yang terdeteksi
            cls_name = model.names.get(int(cls_id), f"Class {int(cls_id)}")
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

        for name, count in class_counts.items():
            info += f"* {name}: `{count}` kali\n"
    
    return plotted_image_rgb, info, detections

# --- FUNGSI KLASIFIKASI SIMULASI (Hanya sebagai ringkasan klasifikasi) ---

def simulate_wall_damage_prediction(image_name):
    """
    Fungsi SIMULASI untuk memberikan hasil klasifikasi 'Crack' atau 'Mould'.
    Ini tetap dipertahankan untuk memberikan hasil kualitatif utama.
    """
    # Menggunakan nama file sebagai seed untuk konsistensi sederhana
    # Menggunakan hash() untuk mendapatkan integer dari nama file
    random.seed(hash(image_name) % 1000)
    
    # Simulasikan persentase kerusakan
    crack_score = random.uniform(0.1, 0.95)
    mould_score = random.uniform(0.1, 0.95)
    
    # Logika sederhana untuk menentukan hasil utama
    if crack_score > 0.7 and crack_score > mould_score:
        result = "Ditemukan **Retak (Crack)** parah. Perbaikan segera diperlukan."
        status_emoji = "🚨"
        status_key = "Crack"
    elif mould_score > 0.7 and mould_score > crack_score:
        result = "Ditemukan **Jamur (Mould)** signifikan. Perlu penanganan kelembaban."
        status_emoji = "🦠"
        status_key = "Mould"
    elif crack_score > 0.5 or mould_score > 0.5:
        # Jika kedua skor di atas 0.5 (kerusakan ganda ringan), kita gabungkan
        if crack_score > 0.5 and mould_score > 0.5:
            result = "Ditemukan **Kerusakan Ganda Ringan** (Retak & Jamur). Monitoring disarankan."
            status_emoji = "⚠️"
            status_key = "Ganda"
        else:
            result = "Ditemukan **Kerusakan Ringan** (Retak/Jamur). Monitoring disarankan."
            status_emoji = "⚠️"
            status_key = "Ringan"
    else:
        result = "**Dinding Tampak Baik (Sehat)**."
        status_emoji = "✅"
        status_key = "Sehat"
        
    detail = (
        f"**Skor Retak (Crack):** `{crack_score:.2f}`\n"
        f"**Skor Jamur (Mould):** `{mould_score:.2f}`"
    )
    
    return status_emoji, result, detail, status_key


# --- FUNGSI UTAMA STREAMLIT ---
def main():
    st.set_page_config(page_title="Deteksi Kerusakan Dinding - Model best.pt", layout="wide")
    
    st.title("🧱 Deteksi Kerusakan Dinding (Real Model)")
    st.markdown("Aplikasi untuk menguji model **YOLOv8 custom (best.pt)** dalam mendeteksi dan melokalisasi kerusakan dinding (retak, jamur, dll).")
    
    # Muat Model
    model = load_yolo_model(MODEL_PATH)
    if model is None:
        # Berhenti jika model gagal dimuat
        st.stop() 

    # --- Sidebar Konfigurasi ---
    st.sidebar.header("⚙️ Konfigurasi Deteksi")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold (Kepercayaan Deteksi)",
        min_value=0.01,
        max_value=1.0,
        value=0.40, # Default lebih tinggi untuk model custom
        step=0.01,
        help="Batas minimum kepercayaan untuk kotak deteksi yang ditampilkan."
    )
    
    # File Uploader
    uploaded_file = st.file_uploader(
        "Unggah Citra Dinding (.jpg, .png)", 
        type=['jpg', 'jpeg', 'png']
    )
    # ---------------------------

    if uploaded_file is not None:
        # Menampilkan Citra Input dan Hasil secara berdampingan
        col1, col2 = st.columns(2)
        
        # Kolom 1: Citra Input
        with col1:
            st.subheader("🖼️ Citra Input")
            uploaded_file.seek(0)
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            uploaded_file.seek(0) # Reset pointer untuk inferensi

        # Kolom 2: Hasil Analisis
        with col2:
            st.subheader("🔬 Hasil Analisis Dinding")
            
            # --- 1. SIMULASI KLASIFIKASI (Tujuan: Menyediakan ringkasan kualitatif) ---
            # Jalankan simulasi utama
            status_emoji, main_result, detail_info, status_key = simulate_wall_damage_prediction(uploaded_file.name)
            
            # Tampilkan hasil simulasi utama
            st.markdown(f"### {status_emoji} {main_result}")
            with st.expander("Lihat Detail Klasifikasi (Simulasi)"):
                st.code(detail_info, language='markdown')

            st.markdown("---")
            st.caption(f"Hasil Deteksi Lokal **{MODEL_PATH}** (Pelokalan Akurat)")
            
            # --- 2. JALANKAN INFERENSI YOLO REAL ---
            uploaded_file.seek(0)
            # Menggunakan run_yolo_inference untuk mendapatkan visualisasi bounding box nyata
            annotated_img, info_yolo, detections = run_yolo_inference(model, uploaded_file, confidence_threshold)
            
            if annotated_img is not None:
                # TAMPILKAN HASIL DENGAN BOUNDING BOX DI SINI
                st.image(annotated_img, caption="Hasil Deteksi Kerusakan dengan Bounding Box", use_container_width=True)
                
                if detections > 0:
                    st.markdown(info_yolo)
                else:
                    st.info("Tidak ada kerusakan (retak/jamur) yang terdeteksi pada batas kepercayaan yang ditentukan.")
            else:
                st.error(info_yolo)
                
    else:
        st.info("⬆️ Silakan unggah citra dinding untuk memulai analisis menggunakan model best.pt.")
        st.markdown("*Aplikasi ini menggunakan model **YOLOv8 custom (best.pt)** untuk pelokalan (localization) dan simulasi klasifikasi kualitatif. Pastikan file `best.pt` tersedia.*")

if __name__ == "__main__":
    main()