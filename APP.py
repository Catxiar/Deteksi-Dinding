import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import time
import io
import csv
import json
import math
import cv2

st.set_page_config(page_title="Deteksi Kerusakan Dinding (OBB)", layout="wide")
st.sidebar.markdown("## Referensi Model")
st.sidebar.markdown(
    """
    Model ini dikembangkan menggunakan **Ultralytics YOLOv8**.
    
    ### Sitasi (BibTeX):
    ```bibtex
    @software{yolov8_ultralytics,
      author = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
      title = {Ultralytics YOLOv8},
      version = {8.0.0},
      year = {2023},
      url = {[https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)},
      orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
      license = {AGPL-3.0}
    }
    ```
    """
)
st.sidebar.markdown("---")

st.markdown(
    """
    <style>
        img {
            max-width: 60% !important;
            height: auto !important;
            margin-left: auto !important;
            margin-right: auto !important;
            display: block !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Sistem Deteksi Kerusakan Dinding (OBB)")
st.subheader("Unggah gambar dan temukan analisis kerusakan retak dan jamur dengan akurat.")

@st.cache_resource
def load_model():
    model_path = r"best.pt"
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Gagal memuat model dari path: {model_path}")
        st.error(f"Error: {e}. Pastikan file model 'best.pt' ada di direktori yang sama.")
        return None

model = load_model()

# --- CALLBACK FUNCTIONS UNTUK SINKRONISASI ---
def sync_conf_slider():
    st.session_state.conf_input = st.session_state.conf_slider
    st.session_state.conf_value = st.session_state.conf_slider

def sync_conf_input():
    st.session_state.conf_slider = st.session_state.conf_input
    st.session_state.conf_value = st.session_state.conf_input

def sync_iou_slider():
    st.session_state.iou_input = st.session_state.iou_slider
    st.session_state.iou_value = st.session_state.iou_slider

def sync_iou_input():
    st.session_state.iou_slider = st.session_state.iou_input
    st.session_state.iou_value = st.session_state.iou_input

def sync_imgsz_slider():
    # Menemukan opsi terdekat untuk number input, karena select_slider memiliki opsi diskrit
    imgsz_options = [320, 480, 640, 800, 1024]
    closest_value = min(imgsz_options, key=lambda x: abs(x - st.session_state.imgsz_slider))
    st.session_state.imgsz_input = closest_value
    st.session_state.imgsz_value = closest_value

def sync_imgsz_input():
    # Menemukan opsi terdekat dari input untuk select slider
    imgsz_options = [320, 480, 640, 800, 1024]
    closest_value = min(imgsz_options, key=lambda x: abs(x - st.session_state.imgsz_input))
    st.session_state.imgsz_slider = closest_value
    st.session_state.imgsz_value = closest_value
# --- MODIFIKASI DIMULAI DI SINI ---
with st.expander("Pengaturan Deteksi Model", expanded=True):
    col_conf, col_iou, col_imgsz = st.columns(3)
    
    # Inisialisasi Session State jika belum ada
    if 'conf_value' not in st.session_state:
        st.session_state['conf_value'] = 0.25
        st.session_state['conf_input'] = 0.25
        st.session_state['conf_slider'] = 0.25
    if 'iou_value' not in st.session_state:
        st.session_state['iou_value'] = 0.45
        st.session_state['iou_input'] = 0.45
        st.session_state['iou_slider'] = 0.45
    if 'imgsz_value' not in st.session_state:
        st.session_state['imgsz_value'] = 480
        st.session_state['imgsz_input'] = 480
        st.session_state['imgsz_slider'] = 480
        
    # 1. CONFIDENCE THRESHOLD
    with col_conf:
        
        # Number Input: Mengatur Slider saat nilai input diubah
        st.number_input(
            "1. Nilai Confidence (Input)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.conf_input,
            step=0.001,
            format="%.3f",
            key="conf_input",
            on_change=sync_conf_input 
        )
        
        # Slider: Mengatur Number Input saat slider digeser
        conf_threshold = st.slider(
            "Batas Keyakinan (Confidence)", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.conf_slider, 
            step=0.001,
            format="%.3f",
            key="conf_slider",
            on_change=sync_conf_slider
        )
        
        # Ambil nilai final yang disinkronkan
        conf_threshold = st.session_state.conf_value

    # 2. IOU THRESHOLD
    with col_iou:
        
        # Number Input
        st.number_input(
            "2. Nilai IOU (Input)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.iou_input,
            step=0.001,
            format="%.3f",
            key="iou_input",
            on_change=sync_iou_input
        )
        
        # Slider
        iou_threshold = st.slider(
            "IOU Threshold (NMS)", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.iou_slider,
            step=0.001,
            format="%.3f",
            key="iou_slider",
            on_change=sync_iou_slider
        )
        
        iou_threshold = st.session_state.iou_value

    # 3. IMGSZ (UKURAN GAMBAR)
    with col_imgsz:
        imgsz_options = [320, 480, 640, 800, 1024]
        
        # Number Input
        st.number_input(
            "3. Ukuran Gambar (Input)",
            min_value=imgsz_options[0],
            max_value=imgsz_options[-1],
            value=st.session_state.imgsz_input,
            step=32, 
            key="imgsz_input",
            on_change=sync_imgsz_input
        )
        
        # Select Slider (hanya menerima opsi diskrit)
        imgsz_choice = st.select_slider(
            "Ukuran Gambar Input (imgsz)",
            options=imgsz_options,
            value=st.session_state.imgsz_slider,
            key="imgsz_slider",
            on_change=sync_imgsz_slider
        )
        
        imgsz_choice = st.session_state.imgsz_value
    
    # KELAS KERUSAKAN 
    selected_class_indices = None
    if model is not None and hasattr(model, 'names'):
        all_classes = list(model.names.values())
        selected_classes = st.multiselect(
            "Pilih Kelas Kerusakan",
            options=all_classes,
            default=all_classes,
        )
        selected_class_indices = [k for k, v in model.names.items() if v in selected_classes]
# --- MODIFIKASI SELESAI DI SINI ---

st.markdown("---")
uploaded_files = st.file_uploader("Unggah Gambar Kerusakan Dinding (jpg/jpeg/png). Unggah banyak gambar untuk analisis trend.", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
st.markdown("---")

def polygon_area(xy):
    if len(xy) < 3:
        return 0.0
    area = 0.0
    n = len(xy)
    for i in range(n):
        x1, y1 = xy[i]
        x2, y2 = xy[(i+1)%n]
        area += x1*y2 - x2*y1
    return abs(area)/2.0

def extract_polygon_from_box(box):
    candidates = ['xy', 'xyxy', 'xywh', 'pts', 'poly', 'points', 'rbox']
    for attr in candidates:
        val = getattr(box, attr, None)
        if val is None:
            continue
        try:
            arr = val.cpu().numpy()
        except Exception:
            try:
                arr = np.array(val)
            except Exception:
                arr = None
        if arr is None:
            continue
        arr = arr.flatten()
        if arr.size == 8:
            pts = [(float(arr[i]), float(arr[i+1])) for i in range(0,8,2)]
            return pts
        if arr.size == 4:
            x1,y1,x2,y2 = arr
            pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
            return pts
    return None

class_severity_base = {
    'crack': 60,
    'mould': 40
}

repair_suggestions = {
    'crack': "Periksa panjang dan kedalaman retak. Rekomendasi: bersihkan area, injeksi epoxy untuk retak dalam, atau patching & repaint. Hubungi profesional jika retak melewati struktur atau lebih dari 5% area.",
    'mould': "Periksa sumber kelembapan. Rekomendasi: pembersihan dengan agen antifungal, perbaiki sumber kebocoran/ventilasi, serta gunakan dehumidifier jika perlu. Gunakan sarung tangan dan ventilasi saat membersihkan.",
}

def create_heatmap_overlay(image_shape, polygons):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        pts_np = np.array(pts, dtype=np.int32)
        cv2.fillPoly(mask, [pts_np], 255)
    
    colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (colored.astype(np.float32) * alpha + np.zeros_like(colored).astype(np.float32) * (1-alpha)).astype(np.uint8), mask

def process_image(file, model, conf_threshold, iou_threshold, imgsz_choice, selected_class_indices):
    image = Image.open(file).convert('RGB')
    img_w, img_h = image.size
    img_np = np.array(image)
    start_time = time.time()
    results = model.predict(
        source=image,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz_choice,
        classes=selected_class_indices
    )
    processing_time = time.time() - start_time

    res_plotted = results[0].plot()
    res_rgb = res_plotted[:, :, ::-1]

    detection_summary = {}
    detection_list = []
    polygons = []
    max_conf = 0.0
    max_conf_cls = 'N/A'

    obb_list = getattr(results[0], 'obb', None)
    boxes_list = getattr(results[0], 'boxes', None)

    if obb_list is not None and len(obb_list) > 0:
        for idx, box in enumerate(obb_list):
            try:
                cls_id = int(box.cls.cpu().numpy()[0])
                conf = float(box.conf.cpu().numpy()[0])
            except Exception:
                try:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                except Exception:
                    continue
            cls_name = model.names.get(cls_id, str(cls_id))

            pts = extract_polygon_from_box(box)
            area_px = 0.0
            if pts is None:
                try:
                    xy = boxes_list[idx].xyxy.cpu().numpy().flatten()
                    x1,y1,x2,y2 = xy
                    pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                except Exception:
                    pts = [(0,0),(0,0),(0,0),(0,0)]
            if pts is not None:
                area_px = polygon_area(pts)
                polygons.append(pts)

            detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1
            detection_list.append({
                'class': cls_name,
                'confidence': conf,
                'area_px': area_px
            })
            if conf > max_conf:
                max_conf = conf
                max_conf_cls = cls_name
    else:
        if boxes_list is not None and len(boxes_list) > 0:
            boxes_arr = boxes_list.xyxy.cpu().numpy()
            cls_arr = boxes_list.cls.cpu().numpy()
            conf_arr = boxes_list.conf.cpu().numpy()
            for i in range(len(boxes_arr)):
                x1,y1,x2,y2 = boxes_arr[i]
                cls_id = int(cls_arr[i])
                conf = float(conf_arr[i])
                cls_name = model.names.get(cls_id, str(cls_id))
                pts = [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]
                area_px = polygon_area(pts)
                polygons.append(pts)
                detection_summary[cls_name] = detection_summary.get(cls_name, 0) + 1
                detection_list.append({
                    'class': cls_name,
                    'confidence': conf,
                    'area_px': area_px
                })
                if conf > max_conf:
                    max_conf = conf
                    max_conf_cls = cls_name

    total_objects = sum(detection_summary.values())

    total_area = sum([d['area_px'] for d in detection_list])
    image_area = img_w * img_h
    area_ratio = (total_area / image_area) if image_area>0 else 0.0

    conf_score = max_conf * 100
    area_score = min(area_ratio * 5000, 100)
    count_score = min(total_objects * 10, 100)

    base = class_severity_base.get(max_conf_cls, 20)
    severity_raw = (0.4 * conf_score) + (0.4 * area_score) + (0.2 * count_score)
    severity = min(max((severity_raw + base) / 2.0, 0), 100)

    if severity >= 71:
        severity_cat = 'Severe'
    elif severity >= 31:
        severity_cat = 'Moderate'
    else:
        severity_cat = 'Minor'

    location_risks = []
    for det in detection_list:
        det_pts = None
        for p in polygons:
            if abs(polygon_area(p) - det['area_px']) < 1e-3:
                det_pts = p
                break
        if det_pts is None:
            location_risks.append('Unknown')
            continue
        cx = sum([pt[0] for pt in det_pts]) / len(det_pts)
        cy = sum([pt[1] for pt in det_pts]) / len(det_pts)
        if cy > 0.7 * img_h:
            location_risks.append('High (near floor)')
        elif cy < 0.3 * img_h:
            location_risks.append('Medium (near ceiling)')
        else:
            location_risks.append('Medium')

    type_severity_notes = {}
    for cls_name, count in detection_summary.items():
        base_val = class_severity_base.get(cls_name, 20)
        if cls_name == 'crack':
            note = 'Retak dapat mengindikasikan isu struktural tergantung kedalaman dan panjang.'
        elif cls_name == 'mould':
            note = 'Mould berhubungan dengan masalah kelembapan dan bisa berdampak kesehatan.'
        else:
            note = ''
        type_severity_notes[cls_name] = {
            'base_severity': base_val,
            'note': note,
            'suggestion': repair_suggestions.get(cls_name, '')
        }

    heatmap_img = None
    mask = None
    if len(polygons) > 0:
        try:
            overlay_colored, mask = create_heatmap_overlay(img_np.shape, polygons)
            heatmap_img = cv2.addWeighted(res_rgb.astype(np.uint8), 0.7, overlay_colored.astype(np.uint8), 0.3, 0)
        except Exception:
            heatmap_img = res_rgb
    else:
        heatmap_img = res_rgb

    report = {
        'file_name': getattr(file, 'name', 'uploaded_image'),
        'image_width': img_w,
        'image_height': img_h,
        'processing_time_s': processing_time,
        'total_objects': total_objects,
        'detection_summary': detection_summary,
        'detection_list': detection_list,
        'total_detected_area_px': total_area,
        'total_detected_area_ratio': area_ratio,
        'severity_score': severity,
        'severity_category': severity_cat,
        'dominant_class': max_conf_cls,
        'dominant_confidence': max_conf,
        'type_severity_notes': type_severity_notes,
        'location_risks': location_risks
    }

    return {
        'image_pil': image,
        'plotted_rgb': res_rgb,
        'heatmap_rgb': heatmap_img,
        'mask': mask,
        'report': report
    }
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    else:
        return obj

if uploaded_files and model is not None:
    results_all = []
    progress_text = st.empty()
    for idx, file in enumerate(uploaded_files):
        progress_text.text(f"Memproses {idx+1}/{len(uploaded_files)}: {getattr(file,'name', 'image')}")
        try:
            res = process_image(file, model, conf_threshold, iou_threshold, imgsz_choice, selected_class_indices)
            results_all.append(res)
        except Exception as e:
            st.error(f"Gagal memproses {getattr(file,'name','image')}: {e}")
    progress_text.empty()

    trend_rows = []
    for r in results_all:
        rep = r['report']
        trend_rows.append({
            'file': rep['file_name'],
            'total_objects': rep['total_objects'],
            'severity_score': rep['severity_score'],
            'area_ratio': rep['total_detected_area_ratio']
        })

    tab1, tab2, tab3, tab4 = st.tabs(["Ringkasan Umum", "Luas & Heatmap", "Trend & Export", "Detail per Gambar"])

    with tab1:
        st.header("Ringkasan Analisis Keseluruhan")
        
        total_images = len(results_all)
        total_objects = sum([r['report']['total_objects'] for r in results_all])
        avg_severity = sum([r['report']['severity_score'] for r in results_all]) / total_images if total_images>0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Gambar Diproses", total_images, help="Jumlah total file gambar yang diunggah dan berhasil diproses.")
        with col2:
            st.metric("Total Deteksi Objek", total_objects, help="Jumlah total kerusakan yang terdeteksi di semua gambar.")
        with col3:
            st.metric("Rata-rata Severity", f"{avg_severity:.1f} / 100", help="Rata-rata skor keparahan kerusakan di semua gambar.")

        st.markdown("---")
        st.subheader("Distribusi Kerusakan per Kelas")
        agg_class = {}
        for r in results_all:
            for k,v in r['report']['detection_summary'].items():
                agg_class[k] = agg_class.get(k,0) + v
        
        class_cols = st.columns(len(agg_class) or 1)
        
        for i, (cls_name, count) in enumerate(agg_class.items()):
            if i < len(class_cols):
                with class_cols[i]:
                    st.info(f"**{cls_name.upper()}**\n\n**{count}** Deteksi")
            else:
                st.info(f"**{cls_name.upper()}**: {count} Deteksi")

    with tab2:
        st.header("Visualisasi Area Kerusakan (Heatmap)")
        options = [r['report']['file_name'] for r in results_all]
        sel = st.selectbox("Pilih gambar untuk preview visualisasi", options=options)
        sel_idx = options.index(sel)
        chosen = results_all[sel_idx]
        rep = chosen['report']
        
        st.subheader(f"Analisis Gambar: {rep['file_name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.image(chosen['plotted_rgb'], caption='Hasil Deteksi (OBB/Bounding Box)', use_container_width=True)
        with col2:
            st.image(chosen['heatmap_rgb'], caption='Heatmap Overlay Area Kerusakan', use_container_width=True)
        
        st.markdown("---")
        st.subheader("Metrik Luas Area")
        col_area_px, col_area_ratio, col_severity = st.columns(3)
        
        with col_area_px:
            st.metric("Luas Area Terdeteksi (px²)", f"{rep['total_detected_area_px']:.2f}")
        with col_area_ratio:
            st.metric("Persentase Area Gambar", f"{rep['total_detected_area_ratio']*100:.3f} %")
        with col_severity:
            st.metric("Severity Score", f"{rep['severity_score']:.1f} / 100", rep['severity_category'])

    with tab3:
        st.header("Analisis Trend dan Ekspor Data")
        
        st.subheader("Data Trend Sederhana")
        
        trend_data_display = [
            {
                'File': row['file'],
                'Objek': row['total_objects'],
                'Severity Score': f"{row['severity_score']:.1f}",
                'Area Ratio (%)': f"{row['area_ratio']*100:.3f}"
            }
            for row in trend_rows
        ]
        st.table(trend_data_display)

        st.markdown("---")
        st.subheader("Opsi Ekspor Data")
        col_csv, col_json = st.columns(2)
        
        csv_buf = io.StringIO()
        csv_writer = csv.writer(csv_buf)
        csv_writer.writerow(['file','total_objects','severity_score','area_ratio'])
        for row in trend_rows:
            csv_writer.writerow([row['file'], row['total_objects'], f"{row['severity_score']:.2f}", f"{row['area_ratio']:.6f}"])
        csv_bytes = csv_buf.getvalue().encode('utf-8')
        
        with col_csv:
            st.download_button('Unduh Trend CSV', data=csv_bytes, file_name=f'trend_analisis_{int(time.time())}.csv', mime='text/csv')
            
        safe_report = make_json_safe([r['report'] for r in results_all])
        json_bytes = json.dumps(safe_report, indent=2).encode('utf-8')

        with col_json:
            st.download_button('Unduh Laporan JSON (Detail)', data=json_bytes, file_name=f'laporan_detail_{int(time.time())}.json', mime='application/json')

    with tab4:
        st.header("Laporan Rinci per Gambar")
        for idx, r in enumerate(results_all):
            rep = r['report']
            with st.expander(f"{rep['file_name']} — Severity: {rep['severity_score']:.1f} ({rep['severity_category']})", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(r['plotted_rgb'], caption='Hasil Deteksi', use_container_width=True)
                with col2:
                    st.image(r['heatmap_rgb'], caption='Heatmap', use_container_width=True)
                
                st.subheader("Statistik Kunci")
                
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                with stat_col1: st.metric("Waktu Proses (s)", f"{rep['processing_time_s']:.2f}")
                with stat_col2: st.metric("Total Objek", rep['total_objects'])
                with stat_col3: st.metric("Dominant Class", rep['dominant_class'], f"Conf: {rep['dominant_confidence']:.3f}")

                st.markdown("---")
                st.subheader("Detail Kerusakan dan Rekomendasi")
                
                for k,v in rep['type_severity_notes'].items():
                    st.markdown(f"**{k.upper()}**")
                    st.write(f"&nbsp;&nbsp;&nbsp;&nbsp;*Catatan:* {v['note']}")
                    st.warning(f"&nbsp;&nbsp;&nbsp;&nbsp;*Saran Perbaikan:* {v['suggestion']}")
                
                st.markdown("---")
                st.subheader("Daftar Objek Terdeteksi")
                for i, d in enumerate(rep['detection_list']):
                    location_risk = rep['location_risks'][i]
                    st.markdown(f"**# {i+1}** | **{d['class']}** - Conf: `{d['confidence']:.3f}` - Area: `{d['area_px']:.2f}` px² - Lokasi Risiko: `{location_risk}`")

                st.markdown("---")
                md = f"# Laporan {rep['file_name']}\n\n"
                md += f"## Ringkasan\n"
                md += f"- Waktu proses: {rep['processing_time_s']:.2f} s\n"
                md += f"- Total objek: {rep['total_objects']}\n"
                md += f"- Luas terdeteksi (px²): {rep['total_detected_area_px']:.2f}\n"
                md += f"- Persentase area: {rep['total_detected_area_ratio']*100:.3f}%\n"
                md += f"- Severity: {rep['severity_score']:.1f} ({rep['severity_category']})\n\n"
                
                md += "## Deteksi per objek\n"
                for i, d in enumerate(rep['detection_list']):
                    location_risk = rep['location_risks'][i]
                    md += f"- {d['class']} ({location_risk}) — conf: {d['confidence']:.3f}, area_px: {d['area_px']:.2f}\n"
                
                st.download_button('Unduh Laporan Markdown (per gambar)', data=md, file_name=f'laporan_{rep['file_name']}_{int(time.time())}.md', mime='text/markdown')

else:
    if model is None:
        st.warning("Model tidak ditemukan. Pastikan path model di kode `app.py` sudah benar.")
    else:
        st.info("Silakan unggah minimal satu gambar untuk memulai deteksi. Gunakan pengaturan model untuk hasil yang lebih optimal.")
