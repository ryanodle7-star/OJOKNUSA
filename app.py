import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import urllib.parse
import urllib.request
import pydeck as pdk
from pathlib import Path


# =========================================================
# KONFIGURASI HALAMAN
# =========================================================

st.set_page_config(
    page_title="OJOKNUSA AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# PATH MODEL
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

LOCAL_MODEL = Path(
    r"C:\OPI AI\model\random_forest_magepanda.pkl"
)

REPO_MODEL = BASE_DIR / "random_forest_magepanda.pkl"

if LOCAL_MODEL.exists():
    MODEL_PATH = LOCAL_MODEL
else:
    MODEL_PATH = REPO_MODEL

# =========================================================
# PATH DATASET
# =========================================================

LOCAL_DATA = Path(
    r"C:\OPI AI\data\raw\Magepanda_synthetic_dataset_corrected.csv"
)

REPO_DATA = BASE_DIR / "Magepanda_synthetic_dataset_corrected.csv"

if LOCAL_DATA.exists():
    DATA_PATH = LOCAL_DATA
else:
    DATA_PATH = REPO_DATA

@st.cache_data
def load_project_data():
    return pd.read_csv(DATA_PATH)

try:
    project_df = load_project_data()
    DATA_LOADED = True
    DATA_ERROR = ""
except Exception as e:
    project_df = pd.DataFrame()
    DATA_LOADED = False
    DATA_ERROR = str(e)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
    MODEL_LOADED = True
    MODEL_ERROR = ""

except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# =========================================================
# FITUR MODEL
# =========================================================

FITUR = [
    "year",
    "latitude",
    "longitude",
    "wave_height_m",
    "wind_speed_ms",
    "elevation_m",
    "distance_to_road_m"
]

TARGET = "change_rate_m_year"


# =========================================================
# CSS
# =========================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #f4f8fb;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #063b4c 0%,
            #075b6e 55%,
            #087f8c 100%
        );
    }

    [data-testid="stSidebar"] {
        color: white;
    }

    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] label {
        color: white !important;
    }

    [data-testid="stSidebar"] button {
        background: rgba(255,255,255,0.96) !important;
        color: #063b4c !important;
        border: 1px solid rgba(255,255,255,0.35) !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
    }

    [data-testid="stSidebar"] button:hover {
        background: #e8f7fa !important;
        color: #063b4c !important;
        border-color: #ffffff !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        color: white;
    }

    .side-brand {
        text-align: center;
        padding: 8px 4px 18px 4px;
    }

    .side-brand-title {
        color: white;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .side-brand-subtitle {
        color: rgba(255,255,255,0.88);
        font-size: 12px;
        line-height: 1.45;
        margin-top: 5px;
    }

    .brand {
        text-align: center;
        padding: 12px 5px 20px 5px;
    }

    .brand-title {
        font-size: 32px;
        font-weight: 800;
        letter-spacing: 1px;
    }

    .brand-subtitle {
        font-size: 13px;
        opacity: 0.85;
    }

    .hero {
        padding: 35px;
        border-radius: 22px;
        background: linear-gradient(
            135deg,
            #063b4c,
            #087f8c,
            #1ca6a8
        );
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }

    .hero h1 {
        font-size: 42px;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        line-height: 1.6;
        opacity: 0.92;
    }

    .card {
        background: #ffffff;
        padding: 24px;
        border-radius: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.07);
        border: 1px solid #e7eef2;
        margin-bottom: 18px;
        color: #173b46;
        line-height: 1.65;
    }
    .card p {
        color: #36545e;
        line-height: 1.65;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .card ul {
        color: #36545e;
        line-height: 1.8;
        padding-left: 22px;
    }
    .home-hero {
        background: linear-gradient(135deg, #063b4c 0%, #087f8c 55%, #1ca6a8 100%);
        color: #ffffff;
        padding: 34px 38px;
        border-radius: 22px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
    }
    .home-hero h1 {
        color: #ffffff !important;
        font-size: 42px;
        line-height: 1.15;
        margin: 0 0 10px 0;
        letter-spacing: 0.5px;
    }
    .home-hero p {
        color: rgba(255,255,255,0.94) !important;
        font-size: 17px;
        line-height: 1.65;
        margin: 7px 0;
    }
    .home-note {
        background: #eaf6f8;
        border-left: 5px solid #087f8c;
        padding: 18px 20px;
        border-radius: 12px;
        color: #164652;
        line-height: 1.65;
        margin-top: 8px;
    }
    .home-note strong {
        color: #063b4c;
    }

    .card-title {
        font-size: 19px;
        font-weight: 700;
        color: #063b4c;
        margin-bottom: 8px;
    }

    .section-title {
        font-size: 30px;
        font-weight: 800;
        color: #063b4c;
        margin-bottom: 20px;
    }

    .metric-box {
        background: white;
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        border: 1px solid #e4edf1;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
    }

    .metric-number {
        font-size: 28px;
        font-weight: 800;
        color: #087f8c;
    }

    .metric-label {
        font-size: 13px;
        color: #60717a;
        margin-top: 4px;
    }

    .risk-high {
        background: #ffe5e5;
        border-left: 6px solid #d9534f;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .risk-medium {
        background: #fff3d6;
        border-left: 6px solid #e0a800;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .risk-low {
        background: #e4f7ec;
        border-left: 6px solid #2e8b57;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }

    .info-box {
        background: #eaf6f8;
        border-left: 5px solid #087f8c;
        padding: 16px;
        border-radius: 10px;
        color: #164652;
        margin: 15px 0;
    }

    .about-card {
        background: #ffffff;
        border: 1px solid #dcebef;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 5px 18px rgba(0,0,0,0.06);
        margin-bottom: 18px;
        color: #173b46;
        line-height: 1.7;
    }

    .about-card h3 {
        color: #063b4c;
        margin: 0 0 10px 0;
        font-size: 20px;
    }

    .about-card p {
        color: #36545e;
        margin: 6px 0;
    }

    .about-card ul {
        color: #36545e;
        line-height: 1.9;
        padding-left: 22px;
        margin-bottom: 0;
    }

    .status-active {
        background: linear-gradient(135deg, #0b7a52, #149b68);
        border: 1px solid #73e0ad;
        border-radius: 14px;
        padding: 13px 14px;
        margin-top: 8px;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.10);
    }

    .status-active-title {
        color: #ffffff !important;
        font-size: 15px;
        font-weight: 800;
        line-height: 1.3;
    }

    .status-active-sub {
        color: #effff7 !important;
        font-size: 11px;
        margin-top: 5px;
    }

    .dashboard-header {
        background: linear-gradient(135deg, #ffffff 0%, #eef8fa 100%);
        border: 1px solid #dbecef;
        border-radius: 18px;
        padding: 22px 24px;
        margin: 8px 0 18px 0;
        box-shadow: 0 5px 18px rgba(0,0,0,0.05);
    }

    .dashboard-header h3 {
        color: #063b4c;
        margin: 0 0 5px 0;
        font-size: 22px;
    }

    .dashboard-header p {
        color: #52636b;
        margin: 0;
        line-height: 1.6;
    }

    .mini-label {
        color: #60717a;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 4px;
    }

    .dashboard-note {
        background: #fffaf0;
        border: 1px solid #f2dfb0;
        border-left: 5px solid #e0a800;
        padding: 15px 18px;
        border-radius: 12px;
        color: #5c4a18;
        line-height: 1.6;
        margin: 12px 0 20px 0;
    }


    footer {
        visibility: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# REFERENSI GARIS PANTAI AKTUAL
# =========================================================

COASTLINE_GEOJSON = BASE_DIR / "data" / "geo" / "GarisPantai_Magepanda_2024.geojson"


@st.cache_data(show_spinner=False)
def load_magepanda_coastline():
    """Utamakan garis pantai Magepanda 2024 dari file lokal.
    Overpass hanya menjadi fallback jika GeoJSON lokal tidak tersedia.
    """
    if COASTLINE_GEOJSON.exists():
        try:
            data = json.loads(COASTLINE_GEOJSON.read_text(encoding="utf-8"))
            paths = []
            for feature in data.get("features", []):
                geom = feature.get("geometry") or {}
                gtype = geom.get("type")
                coords = geom.get("coordinates", [])
                if gtype == "LineString" and len(coords) >= 2:
                    paths.append({"path": [[float(x), float(y)] for x, y in coords]})
                elif gtype == "MultiLineString":
                    for line in coords:
                        if len(line) >= 2:
                            paths.append({"path": [[float(x), float(y)] for x, y in line]})
            if paths:
                return paths, "GarisPantai_Magepanda_2024.geojson (lokal)"
            local_error = "GeoJSON lokal tidak berisi geometri garis yang valid."
        except Exception as exc:
            local_error = f"GeoJSON lokal gagal dibaca: {exc}"
    else:
        local_error = "GeoJSON garis pantai lokal belum tersedia."

    query = (
        '[out:json][timeout:25];'
        'way["natural"="coastline"]'
        '(around:12000,-8.5487,122.0486);'
        'out geom;'
    )
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_error = None
    for endpoint in endpoints:
        try:
            payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
            request = urllib.request.Request(
                endpoint, data=payload,
                headers={"User-Agent": "OJOKNUSA-AI/1.0"}, method="POST"
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
            paths = []
            for element in data.get("elements", []):
                geometry = element.get("geometry", [])
                path = [[point["lon"], point["lat"]] for point in geometry]
                if len(path) >= 2:
                    paths.append({"path": path})
            if paths:
                return paths, "OpenStreetMap / Overpass API (fallback)"
        except Exception as exc:
            last_error = str(exc)

    return [], last_error or local_error


@st.cache_data(ttl=86400, show_spinner=False)
def build_coastal_display_coordinates(coastline_paths, n_points, seed=42):
    """Membuat koordinat DISPLAY sintetis yang mengikuti geometri garis pantai.

    Koordinat asli dataset tidak diubah. Fungsi ini hanya membuat posisi
    visualisasi baru berdasarkan geometri coastline dari OpenStreetMap.
    Versi ini sengaja tidak memakai ambang panjang 500 m per way, karena
    coastline OSM sering tersusun dari banyak segmen pendek.
    """
    if not coastline_paths or n_points <= 0:
        return None

    ref_lat = MAGEPANDA_LAT
    lat_scale = 111320.0
    lon_scale = 111320.0 * np.cos(np.deg2rad(ref_lat))

    # Ubah semua way coastline menjadi koordinat lokal (meter).
    paths = []
    lengths = []
    for item in coastline_paths:
        raw_path = item.get("path", [])
        if len(raw_path) < 2:
            continue

        xy = np.asarray(
            [
                [
                    (float(lon) - MAGEPANDA_LON) * lon_scale,
                    (float(lat) - ref_lat) * lat_scale,
                ]
                for lon, lat in raw_path
            ],
            dtype=float,
        )

        # Buang titik berulang/segmen nol panjang.
        if len(xy) >= 2:
            keep = np.r_[True, np.any(np.diff(xy, axis=0) != 0, axis=1)]
            xy = xy[keep]

        if len(xy) < 2:
            continue

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        length = float(seg.sum())
        if length > 20:  # tetap menerima way pendek yang merupakan bagian coastline
            paths.append(xy)
            lengths.append(length)

    if not paths:
        return None

    # Jika OSM mengembalikan banyak potongan sangat kecil, tetap gunakan
    # geometri tersebut, tetapi alokasikan titik berdasarkan panjang garis.
    total_length = float(sum(lengths))
    if total_length <= 0:
        return None

    raw_counts = np.asarray(lengths, dtype=float) / total_length * n_points
    counts = np.floor(raw_counts).astype(int)
    counts = np.maximum(counts, 0)

    # Pastikan segmen yang mendapat porsi kecil tetap memperoleh titik.
    fractional_order = np.argsort(-(raw_counts - counts))
    remaining = n_points - int(counts.sum())
    for idx in fractional_order[:max(0, remaining)]:
        counts[idx] += 1

    # Jika rounding masih menyisakan selisih, distribusikan ke segmen terpanjang.
    while counts.sum() < n_points:
        counts[int(np.argmax(lengths))] += 1
    while counts.sum() > n_points:
        candidates = np.where(counts > 0)[0]
        if len(candidates) == 0:
            return None
        idx = candidates[int(np.argmax(counts[candidates]))]
        counts[idx] -= 1

    rng = np.random.default_rng(seed)
    result = []

    for path_idx, (xy, k) in enumerate(zip(paths, counts)):
        if k <= 0 or len(xy) < 2:
            continue

        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(seg)]
        total = float(cumulative[-1])
        if total <= 0:
            continue

        # Hindari titik yang semuanya persis berada pada ujung-ujung way.
        if k == 1:
            targets = np.array([total * 0.5])
        else:
            spacing = total / k
            targets = (np.arange(k) + 0.5) * spacing

        for local_idx, target in enumerate(targets):
            idx = int(np.searchsorted(cumulative, target, side="right") - 1)
            idx = max(0, min(idx, len(xy) - 2))

            denom = cumulative[idx + 1] - cumulative[idx]
            frac = 0.0 if denom <= 0 else (target - cumulative[idx]) / denom
            p = xy[idx] + frac * (xy[idx + 1] - xy[idx])

            # Tangent lokal mengikuti bentuk coastline.
            if idx == 0:
                tangent = xy[1] - xy[0]
            elif idx >= len(xy) - 2:
                tangent = xy[-1] - xy[-2]
            else:
                tangent = xy[idx + 1] - xy[idx - 1]

            norm = float(np.linalg.norm(tangent))
            if norm <= 0:
                continue
            tangent = tangent / norm

            # Offset sangat kecil dari garis pantai agar marker mudah dilihat.
            normal = np.array([-tangent[1], tangent[0]])
            side = -1.0 if (local_idx + path_idx) % 2 else 1.0
            offset_m = float(rng.uniform(15, 70)) * side
            along_jitter = float(rng.uniform(-20, 20))
            display_xy = p + normal * offset_m + tangent * along_jitter

            display_lon = MAGEPANDA_LON + display_xy[0] / lon_scale
            display_lat = ref_lat + display_xy[1] / lat_scale
            result.append((display_lat, display_lon))

    if len(result) < n_points:
        return None

    return result[:n_points]


# =========================================================
# SESSION STATE
# =========================================================

if "menu" not in st.session_state:
    st.session_state.menu = "Beranda"


# =========================================================
# FUNGSI KLASIFIKASI
# =========================================================

def klasifikasi_perubahan(prediksi):

    if prediksi < -0.5:
        return "Abrasi"

    elif prediksi <= 0.5:
        return "Stabil"

    else:
        return "Akresi"


# =========================================================
# FUNGSI RISIKO
# =========================================================

def hitung_risiko(prediksi, distance_to_road):

    # Skor berdasarkan kecenderungan abrasi

    if prediksi < -1:
        skor_abrasi = 3

    elif prediksi < -0.5:
        skor_abrasi = 2

    elif prediksi < 0:
        skor_abrasi = 1

    else:
        skor_abrasi = 0


    # Skor berdasarkan jarak jalan

    if distance_to_road <= 100:
        skor_jalan = 3

    elif distance_to_road <= 300:
        skor_jalan = 2

    else:
        skor_jalan = 1


    total = skor_abrasi + skor_jalan


    if total >= 5:
        risiko = "Tinggi"

    elif total >= 3:
        risiko = "Sedang"

    else:
        risiko = "Rendah"


    return risiko, total


# =========================================================
# REFERENSI LOKASI STUDI
# =========================================================

# Koordinat referensi Magepanda dari sumber geospasial publik.
# Ini hanya digunakan sebagai titik referensi peta, bukan sebagai
# pengganti koordinat observasi/data sintetis.
MAGEPANDA_LAT = -8.5487
MAGEPANDA_LON = 122.0486

def jarak_ke_magepanda_km(lat, lon):
    lat1 = np.radians(MAGEPANDA_LAT)
    lon1 = np.radians(MAGEPANDA_LON)
    lat2 = np.radians(lat)
    lon2 = np.radians(lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2) + (
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="side-brand">
            <div class="side-brand-title">🌊 OJOKNUSA</div>
            <div class="side-brand-subtitle">
                Artificial Intelligence<br>
                for Coastal Analysis
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("<h3 style='color:white; margin-bottom:10px;'>MENU</h3>", unsafe_allow_html=True)


    if st.button(
        "🏠  Beranda",
        use_container_width=True
    ):
        st.session_state.menu = "Beranda"


    if st.button(
        "🔮  Prediksi Manual",
        use_container_width=True
    ):
        st.session_state.menu = "Prediksi Manual"


    if st.button(
        "📂  Analisis CSV",
        use_container_width=True
    ):
        st.session_state.menu = "Analisis CSV"


    if st.button(
        "🗺️  Peta Pesisir",
        use_container_width=True
    ):
        st.session_state.menu = "Peta Pesisir"


    if st.button(
        "ℹ️  Tentang OJOKNUSA",
        use_container_width=True
    ):
        st.session_state.menu = "Tentang OJOKNUSA"


    st.markdown("---")


    if MODEL_LOADED:
        st.markdown(
            '<div class="status-active">'
            '<div class="status-active-title">🟢 Model AI Aktif</div>'
            '<div class="status-active-sub">Random Forest siap digunakan</div>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="background:rgba(210, 45, 45, 0.22); border:1px solid rgba(255, 130, 130, 0.45); border-radius:12px; padding:13px 14px; margin-top:8px; color:#ffffff;">'
            '<div style="font-size:15px; font-weight:700; color:#ffffff;">🔴&nbsp; Model Tidak Ditemukan</div>'
            '<div style="font-size:11px; color:#ffe2e2; margin-top:4px;">Periksa file model</div>'
            '</div>',
            unsafe_allow_html=True
        )


    st.markdown(
        """
        <div style="
            font-size:12px;
            opacity:0.75;
            text-align:center;
        ">

        OJOKNUSA AI<br>
        Prototype Research Dashboard

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# BERANDA
# =========================================================

if st.session_state.menu == "Beranda":

    st.markdown(
        """
        <div class="hero">
            <h1>🌊 OJOKNUSA AI</h1>
            <p>
            Platform prototype berbasis Artificial Intelligence untuk
            pemodelan perubahan garis pantai dan analisis awal risiko
            abrasi terhadap infrastruktur pesisir.
            </p>
            <p><b>Space + Time + AI</b> untuk membaca dinamika pesisir.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Pusat Analisis Pesisir</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='color:#52636b; font-size:16px; margin-top:-10px;'>"
        "Satu dashboard untuk data, prediksi, analisis risiko, dan visualisasi peta."
        "</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        st.markdown(
            """<div class="card"><div class="card-title">🌊 Dinamika Garis Pantai</div>
            <p>Menganalisis kecenderungan perubahan garis pantai menggunakan informasi spasial dan temporal.</p></div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """<div class="card"><div class="card-title">🤖 Artificial Intelligence</div>
            <p>Random Forest Regressor digunakan untuk memperkirakan laju perubahan garis pantai per tahun.</p></div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """<div class="card"><div class="card-title">🛣️ Risiko Infrastruktur</div>
            <p>Hasil prediksi dapat dikombinasikan dengan jarak ke jalan sebagai analisis risiko prototype.</p></div>""",
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🔬 Parameter Model</div>
            <p>
            Model menggunakan tahun, latitude, longitude, tinggi gelombang,
            kecepatan angin, elevasi, dan jarak terhadap jalan.
            </p>
            <div class="info-box">
            ⚠️ <b>Status penelitian:</b> dataset yang digunakan pada tahap ini
            adalah <b>sintetis/purwarupa</b>. Hasil aplikasi belum boleh dianggap
            sebagai pengukuran lapangan atau validasi kondisi pantai sebenarnya.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-header">
            <h3>📊 Ringkasan Dashboard</h3>
            <p>
            Ringkasan otomatis dari dataset proyek dan hasil prediksi model.
            Gunakan bagian ini sebagai overview sebelum masuk ke analisis detail.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not DATA_LOADED:
        st.error("Dashboard belum dapat dihitung karena dataset proyek tidak berhasil dimuat.")
        st.code(DATA_ERROR)
    else:
        dashboard_df = project_df.copy()

        # Pastikan kolom numerik yang dibutuhkan tersedia.
        dashboard_numeric = [
            "year",
            "latitude",
            "longitude",
            "wave_height_m",
            "wind_speed_ms",
            "elevation_m",
            "distance_to_road_m",
            TARGET,
        ]

        missing_dashboard = [
            c for c in dashboard_numeric
            if c not in dashboard_df.columns
        ]

        if missing_dashboard:
            st.warning(
                "Beberapa kolom dashboard belum tersedia: "
                + ", ".join(missing_dashboard)
            )
        else:
            for col in dashboard_numeric:
                dashboard_df[col] = pd.to_numeric(
                    dashboard_df[col], errors="coerce"
                )

            dashboard_df = dashboard_df.dropna(
                subset=["year", TARGET]
            ).copy()

            # -----------------------------------------------------
            # KPI UTAMA
            # -----------------------------------------------------
            total_rows = len(dashboard_df)
            total_years = dashboard_df["year"].nunique()
            mean_change = dashboard_df[TARGET].mean()
            median_road = dashboard_df["distance_to_road_m"].median()

            k1, k2, k3, k4 = st.columns(4, gap="medium")

            with k1:
                st.metric(
                    "📦 Data",
                    f"{total_rows:,}",
                    help="Jumlah baris data yang terbaca dari dataset proyek."
                )

            with k2:
                st.metric(
                    "📅 Periode",
                    f"{total_years} tahun",
                    help="Jumlah tahun unik yang tersedia pada dataset."
                )

            with k3:
                st.metric(
                    "🌊 Rata-rata perubahan",
                    f"{mean_change:.3f} m/th",
                    help=f"Rata-rata kolom {TARGET} pada dataset proyek."
                )

            with k4:
                if pd.isna(median_road):
                    road_value = "—"
                else:
                    road_value = f"{median_road:.1f} m"
                st.metric(
                    "🛣️ Median jarak jalan",
                    road_value,
                    help="Median jarak titik data terhadap jalan."
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # -----------------------------------------------------
            # TREN PERUBAHAN
            # -----------------------------------------------------
            left, right = st.columns([1.45, 1], gap="large")

            with left:
                st.markdown(
                    '<div class="card-title">📈 Tren Perubahan Garis Pantai</div>',
                    unsafe_allow_html=True
                )

                trend = (
                    dashboard_df
                    .groupby("year", as_index=True)[TARGET]
                    .mean()
                    .sort_index()
                )

                if len(trend):
                    st.line_chart(trend)
                    st.caption(
                        "Rata-rata perubahan per tahun berdasarkan kolom "
                        f"`{TARGET}` pada dataset proyek."
                    )
                else:
                    st.info("Belum ada data tren yang dapat ditampilkan.")

            # -----------------------------------------------------
            # DISTRIBUSI PREDIKSI AI
            # -----------------------------------------------------
            with right:
                st.markdown(
                    '<div class="card-title">🤖 Distribusi Kelas Prediksi AI</div>',
                    unsafe_allow_html=True
                )

                if MODEL_LOADED:
                    pred_df = dashboard_df.copy()
                    pred_features = [
                        c for c in FITUR
                        if c in pred_df.columns
                    ]

                    if len(pred_features) == len(FITUR):
                        valid_pred = pred_df.dropna(
                            subset=FITUR
                        ).copy()

                        if len(valid_pred):
                            valid_pred["prediksi_dashboard"] = model.predict(
                                valid_pred[FITUR]
                            )
                            valid_pred["kelas_dashboard"] = (
                                valid_pred["prediksi_dashboard"]
                                .apply(klasifikasi_perubahan)
                            )

                            kelas_order = ["Abrasi", "Stabil", "Akresi"]
                            kelas_dist = (
                                valid_pred["kelas_dashboard"]
                                .value_counts()
                                .reindex(kelas_order, fill_value=0)
                            )

                            st.bar_chart(kelas_dist)
                            st.caption(
                                "Distribusi ini berasal dari prediksi model "
                                "pada data yang tersedia."
                            )
                        else:
                            st.info(
                                "Tidak ada baris lengkap untuk menjalankan prediksi AI."
                            )
                    else:
                        st.info("Kolom fitur model belum lengkap.")
                else:
                    st.warning(
                        "Model AI tidak aktif, sehingga distribusi prediksi "
                        "belum dapat dihitung."
                    )

            # -----------------------------------------------------
            # TARGET VS PREDIKSI PER TAHUN
            # -----------------------------------------------------
            if MODEL_LOADED:
                pred_compare = dashboard_df.copy()
                if all(c in pred_compare.columns for c in FITUR):
                    pred_compare = pred_compare.dropna(
                        subset=FITUR
                    ).copy()

                    if len(pred_compare):
                        pred_compare["prediksi_dashboard"] = model.predict(
                            pred_compare[FITUR]
                        )

                        compare_year = (
                            pred_compare
                            .groupby("year")[[TARGET, "prediksi_dashboard"]]
                            .mean()
                            .sort_index()
                            .rename(
                                columns={
                                    TARGET: "Target dataset",
                                    "prediksi_dashboard": "Prediksi AI",
                                }
                            )
                        )

                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(
                            '<div class="card-title">🎯 Target Dataset vs Prediksi AI</div>',
                            unsafe_allow_html=True
                        )
                        st.line_chart(compare_year)
                        st.caption(
                            "Perbandingan agregat tahunan; ini bukan validasi "
                            "akurasi model pada data observasi lapangan."
                        )

            # -----------------------------------------------------
            # RISIKO PROTOTYPE
            # -----------------------------------------------------
            if MODEL_LOADED and all(c in dashboard_df.columns for c in FITUR):
                risk_df = dashboard_df.dropna(
                    subset=FITUR
                ).copy()

                if len(risk_df):
                    risk_df["prediksi_dashboard"] = model.predict(
                        risk_df[FITUR]
                    )

                    risk_df["risiko_dashboard"] = risk_df.apply(
                        lambda row: hitung_risiko(
                            row["prediksi_dashboard"],
                            row["distance_to_road_m"],
                        )[0],
                        axis=1,
                    )

                    st.markdown("<br>", unsafe_allow_html=True)

                    r1, r2 = st.columns([1, 1.2], gap="large")

                    with r1:
                        st.markdown(
                            '<div class="card-title">⚠️ Distribusi Risiko Prototype</div>',
                            unsafe_allow_html=True
                        )
                        risk_order = ["Rendah", "Sedang", "Tinggi"]
                        risk_dist = (
                            risk_df["risiko_dashboard"]
                            .value_counts()
                            .reindex(risk_order, fill_value=0)
                        )
                        st.bar_chart(risk_dist)

                    with r2:
                        st.markdown(
                            '<div class="card-title">🧭 Interpretasi Ringkas</div>',
                            unsafe_allow_html=True
                        )
                        st.markdown(
                            """
                            <div class="dashboard-note">
                            <b>Perlu diperhatikan:</b><br>
                            Kategori risiko pada dashboard ini merupakan
                            aturan <b>prototype</b> yang menggabungkan
                            kecenderungan perubahan dan jarak terhadap jalan.
                            Kategori tersebut belum merupakan indeks risiko
                            ilmiah yang tervalidasi.
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.markdown(
                """
                <div class="dashboard-note">
                ⚠️ <b>Status data:</b> dataset pada tahap ini merupakan
                <b>sintetis/purwarupa</b>. Ringkasan dan visualisasi di atas
                digunakan untuk demonstrasi alur analisis OJOKNUSA AI dan
                belum boleh dianggap sebagai pengukuran kondisi pantai aktual.
                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# PREDIKSI MANUAL
# =========================================================

elif st.session_state.menu == "Prediksi Manual":

    st.markdown(
        '<div class="section-title">🔮 Prediksi Perubahan Garis Pantai</div>',
        unsafe_allow_html=True
    )


    if not MODEL_LOADED:

        st.error(
            "Model AI tidak berhasil dimuat."
        )

        st.code(
            MODEL_ERROR
        )


    else:

        st.markdown(
            """
            <div class="card">

            Masukkan kondisi lokasi pesisir.
            OJOKNUSA akan menghitung estimasi
            <b>perubahan garis pantai per tahun</b>.

            </div>
            """,
            unsafe_allow_html=True
        )


        col1, col2 = st.columns(2)


        with col1:

            year = st.number_input(
                "📅 Tahun",
                min_value=2000,
                max_value=2100,
                value=2026,
                step=1
            )


            latitude = st.number_input(
                "📍 Latitude",
                value=-8.50,
                format="%.6f"
            )


            longitude = st.number_input(
                "📍 Longitude",
                value=122.00,
                format="%.6f"
            )


            wave_height = st.number_input(
                "🌊 Tinggi gelombang (m)",
                min_value=0.0,
                value=1.5,
                step=0.1
            )


        with col2:

            wind_speed = st.number_input(
                "💨 Kecepatan angin (m/s)",
                min_value=0.0,
                value=8.0,
                step=0.5
            )


            elevation = st.number_input(
                "⛰️ Elevasi (m)",
                value=2.0,
                step=0.5
            )


            distance_to_road = st.number_input(
                "🛣️ Jarak ke jalan (m)",
                min_value=0.0,
                value=50.0,
                step=10.0
            )


        st.markdown(
            "<br>",
            unsafe_allow_html=True
        )


        if st.button(
            "🚀 Jalankan Prediksi",
            type="primary",
            use_container_width=True
        ):

            data_input = pd.DataFrame(
                [{
                    "year": year,
                    "latitude": latitude,
                    "longitude": longitude,
                    "wave_height_m": wave_height,
                    "wind_speed_ms": wind_speed,
                    "elevation_m": elevation,
                    "distance_to_road_m": distance_to_road
                }]
            )


            try:

                prediksi = model.predict(
                    data_input[FITUR]
                )[0]


                kelas = klasifikasi_perubahan(
                    prediksi
                )


                risiko, skor = hitung_risiko(
                    prediksi,
                    distance_to_road
                )


                st.markdown("---")


                c1, c2, c3 = st.columns(3)


                with c1:

                    st.metric(
                        "Perubahan garis pantai",
                        f"{prediksi:.3f} m/tahun"
                    )


                with c2:

                    st.metric(
                        "Kelas perubahan",
                        kelas
                    )


                with c3:

                    st.metric(
                        "Risiko prototype",
                        risiko
                    )


                if kelas == "Abrasi":

                    st.error(
                        f"🔴 Prediksi menunjukkan kecenderungan "
                        f"**abrasi** sebesar {abs(prediksi):.3f} m/tahun."
                    )


                elif kelas == "Akresi":

                    st.success(
                        f"🟢 Prediksi menunjukkan kecenderungan "
                        f"**akresi** sebesar {prediksi:.3f} m/tahun."
                    )


                else:

                    st.info(
                        "🟡 Prediksi menunjukkan kondisi relatif stabil."
                    )


                if risiko == "Tinggi":

                    st.markdown(
                        f"""
                        <div class="risk-high">

                        <b>🔴 Risiko prototype: TINGGI</b><br>

                        Skor risiko: {skor}/6

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                elif risiko == "Sedang":

                    st.markdown(
                        f"""
                        <div class="risk-medium">

                        <b>🟡 Risiko prototype: SEDANG</b><br>

                        Skor risiko: {skor}/6

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                else:

                    st.markdown(
                        f"""
                        <div class="risk-low">

                        <b>🟢 Risiko prototype: RENDAH</b><br>

                        Skor risiko: {skor}/6

                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                st.caption(
                    "Kategori risiko di atas merupakan aturan "
                    "prototype dan belum merupakan indeks risiko "
                    "ilmiah tervalidasi."
                )


            except Exception as e:

                st.error(
                    f"Prediksi gagal dilakukan: {e}"
                )


# =========================================================
# ANALISIS CSV
# =========================================================

elif st.session_state.menu == "Analisis CSV":

    st.markdown(
        '<div class="section-title">📂 Analisis Dataset CSV</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        """
        <div class="card">

        Upload file CSV yang memiliki kolom input model.
        OJOKNUSA akan melakukan prediksi secara otomatis
        untuk seluruh baris.

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Pilih file CSV",
        type=["csv"],
        key="analysis_csv"
    )


    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file
            )


            st.subheader(
                "📋 Data yang diupload"
            )


            st.dataframe(
                df,
                use_container_width=True
            )


            kolom_hilang = [
                kolom
                for kolom in FITUR
                if kolom not in df.columns
            ]


            if len(kolom_hilang) > 0:

                st.error(
                    "Kolom berikut belum tersedia:"
                )

                st.write(
                    kolom_hilang
                )


            elif not MODEL_LOADED:

                st.error(
                    "Model AI tidak berhasil dimuat."
                )


            else:

                if st.button(
                    "🚀 Jalankan Prediksi CSV",
                    type="primary",
                    use_container_width=True
                ):

                    hasil = df.copy()


                    hasil[
                        "prediksi_change_rate_m_year"
                    ] = model.predict(
                        hasil[FITUR]
                    )


                    hasil[
                        "kelas_prediksi"
                    ] = hasil[
                        "prediksi_change_rate_m_year"
                    ].apply(
                        klasifikasi_perubahan
                    )


                    hasil[
                        "risiko_prototype"
                    ] = hasil.apply(
                        lambda row: hitung_risiko(
                            row[
                                "prediksi_change_rate_m_year"
                            ],
                            row[
                                "distance_to_road_m"
                            ]
                        )[0],
                        axis=1
                    )


                    st.success(
                        "Prediksi berhasil dilakukan."
                    )


                    st.subheader(
                        "📊 Hasil Prediksi"
                    )


                    st.dataframe(
                        hasil,
                        use_container_width=True
                    )


                    st.subheader(
                        "📈 Distribusi Kelas Prediksi"
                    )


                    distribusi = (
                        hasil[
                            "kelas_prediksi"
                        ].value_counts()
                    )


                    st.bar_chart(
                        distribusi
                    )


                    st.subheader(
                        "⚠️ Distribusi Risiko Prototype"
                    )


                    risiko_dist = (
                        hasil[
                            "risiko_prototype"
                        ].value_counts()
                    )


                    st.bar_chart(
                        risiko_dist
                    )


                    csv_hasil = hasil.to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    )


                    st.download_button(
                        "⬇️ Download hasil prediksi",
                        data=csv_hasil,
                        file_name="hasil_prediksi_ojoknusa.csv",
                        mime="text/csv",
                        use_container_width=True
                    )


        except Exception as e:

            st.error(
                f"File CSV tidak dapat diproses: {e}"
            )


# =========================================================
# PETA PESISIR
# =========================================================

elif st.session_state.menu == "Peta Pesisir":

    st.markdown(
        '<div class="section-title">🗺️ Peta Risiko Abrasi</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="card">
            <div class="card-title">🌊 OJOKNUSA Coastal Risk Map</div>
            <p>
            Titik pada peta diberi warna berdasarkan hasil prediksi AI.
            Ukuran titik mengikuti skor risiko prototype.
            </p>
            <p>
            <b>Catatan:</b> dataset masih sintetis/purwarupa.
            Peta ini belum merupakan peta risiko pantai aktual.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not DATA_LOADED:
        st.error("Dataset proyek tidak berhasil dimuat.")
        st.code(DATA_ERROR)
    elif not MODEL_LOADED:
        st.error("Model AI tidak berhasil dimuat.")
        st.code(MODEL_ERROR)
    else:
        map_df = project_df.copy()
        required_columns = [
            "year", "latitude", "longitude",
            "wave_height_m", "wind_speed_ms",
            "elevation_m", "distance_to_road_m"
        ]
        missing_columns = [c for c in required_columns if c not in map_df.columns]

        if missing_columns:
            st.error("Kolom dataset belum lengkap: " + ", ".join(missing_columns))
        else:
            for col in required_columns:
                map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
            map_df = map_df.dropna(subset=["latitude", "longitude"]).copy()

            years = sorted(map_df["year"].dropna().unique())
            if years:
                selected_year = st.selectbox(
                    "📅 Tahun analisis",
                    years,
                    index=len(years) - 1
                )
                map_df = map_df[map_df["year"] == selected_year].copy()
            else:
                selected_year = None

            if len(map_df) == 0:
                st.warning("Tidak ada titik valid pada tahun yang dipilih.")
            else:
                map_df["prediksi_change_rate_m_year"] = model.predict(map_df[FITUR])
                map_df["kelas_prediksi"] = map_df["prediksi_change_rate_m_year"].apply(klasifikasi_perubahan)
                risk_pairs = map_df.apply(
                    lambda r: hitung_risiko(
                        r["prediksi_change_rate_m_year"],
                        r["distance_to_road_m"]
                    ),
                    axis=1
                )
                map_df["risiko_prototype"] = [x[0] for x in risk_pairs]
                map_df["skor_risiko"] = [x[1] for x in risk_pairs]

                # Validasi posisi dataset terhadap pusat referensi Magepanda.
                map_df["jarak_ke_magepanda_km"] = jarak_ke_magepanda_km(
                    map_df["latitude"].to_numpy(),
                    map_df["longitude"].to_numpy()
                )
                median_distance = float(map_df["jarak_ke_magepanda_km"].median())

                if median_distance > 15:
                    st.warning(
                        f"⚠️ Dataset sintetis saat ini memiliki median jarak "
                        f"{median_distance:.1f} km dari titik referensi Magepanda. "
                        "Koordinat tidak saya geser otomatis karena itu akan "
                        "mengubah data penelitian. Dataset perlu dikoreksi "
                        "dan model kemudian dilatih ulang."
                    )
                else:
                    st.success(
                        f"📍 Posisi dataset berada di sekitar zona referensi "
                        f"Magepanda (median {median_distance:.1f} km)."
                    )

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("📍 Jumlah titik", f"{len(map_df):,}")
                with c2:
                    st.metric("🔴 Abrasi", f"{(map_df['kelas_prediksi'] == 'Abrasi').sum():,}")
                with c3:
                    st.metric("🟡 Stabil", f"{(map_df['kelas_prediksi'] == 'Stabil').sum():,}")
                with c4:
                    st.metric("🟢 Akresi", f"{(map_df['kelas_prediksi'] == 'Akresi').sum():,}")

                st.subheader(f"🌊 Sebaran Prediksi {selected_year}")

                # Titik referensi Magepanda.
                st.caption(
                    f"Titik referensi Magepanda: {MAGEPANDA_LAT:.4f}, {MAGEPANDA_LON:.4f}. "
                    "Koordinat ini hanya sebagai referensi lokasi peta."
                )

                # ---------------------------------------------------------
                # PETA BERBASIS GEOMETRI GARIS PANTAI - V4
                # ---------------------------------------------------------
                # Garis pantai referensi diprioritaskan dari GeoJSON lokal Magepanda 2024.
                # Koordinat asli dataset tetap disimpan utuh.
                # Untuk VISUALISASI saja, titik sintetis dibuat mengikuti garis
                # pantai sehingga distribusinya tidak lagi membentuk diagonal.
                coastline_paths, coastline_source = load_magepanda_coastline()

                display_coords = None
                if coastline_paths:
                    display_coords = build_coastal_display_coordinates(
                        coastline_paths,
                        len(map_df),
                        seed=int(selected_year) if selected_year is not None else 42,
                    )

                if display_coords is not None:
                    map_df["display_latitude"] = [p[0] for p in display_coords]
                    map_df["display_longitude"] = [p[1] for p in display_coords]
                    map_df["koordinat_peta"] = "Sintetis mengikuti garis pantai (visualisasi)"
                else:
                    map_df["display_latitude"] = map_df["latitude"]
                    map_df["display_longitude"] = map_df["longitude"]
                    map_df["koordinat_peta"] = "Koordinat dataset asli"

                # Radius marker dalam meter; risiko lebih tinggi dibuat lebih menonjol.
                radius_map = {"Rendah": 45, "Sedang": 75, "Tinggi": 105}
                map_df["radius_m"] = map_df["risiko_prototype"].map(radius_map).fillna(50)

                def make_point_layer(data, kelas, color):
                    subset = data[data["kelas_prediksi"] == kelas].copy()
                    if subset.empty:
                        return None
                    subset["tooltip_label"] = subset.apply(
                        lambda r: (
                            f"Segmen: {r.get('segment_id', '-') } | "
                            f"Kelas: {r['kelas_prediksi']} | "
                            f"Perubahan: {r['prediksi_change_rate_m_year']:.3f} m/tahun | "
                            f"Risiko: {r['risiko_prototype']}"
                        ),
                        axis=1,
                    )
                    return pdk.Layer(
                        "ScatterplotLayer",
                        data=subset,
                        get_position="[display_longitude, display_latitude]",
                        get_radius="radius_m",
                        get_fill_color=color,
                        get_line_color=[255, 255, 255, 230],
                        get_line_width=1,
                        stroked=True,
                        filled=True,
                        pickable=True,
                        radius_min_pixels=4,
                        radius_max_pixels=12,
                    )

                layers = []

                if coastline_paths:
                    layers.append(
                        pdk.Layer(
                            "PathLayer",
                            data=coastline_paths,
                            get_path="path",
                            get_color=[0, 112, 130, 220],
                            get_width=5,
                            width_min_pixels=2,
                            pickable=False,
                        )
                    )

                class_colors = {
                    "Abrasi": [217, 83, 79, 220],
                    "Stabil": [240, 173, 78, 220],
                    "Akresi": [46, 139, 87, 220],
                }
                for kelas, color in class_colors.items():
                    layer = make_point_layer(map_df, kelas, color)
                    if layer is not None:
                        layers.append(layer)

                reference_df = pd.DataFrame({
                    "longitude": [MAGEPANDA_LON],
                    "latitude": [MAGEPANDA_LAT],
                })
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=reference_df,
                        get_position="[longitude, latitude]",
                        get_radius=140,
                        get_fill_color=[255, 255, 255, 255],
                        get_line_color=[6, 59, 76, 255],
                        get_line_width=3,
                        stroked=True,
                        filled=True,
                        pickable=False,
                        radius_min_pixels=7,
                        radius_max_pixels=12,
                    )
                )

                view = pdk.ViewState(
                    latitude=float(map_df["display_latitude"].mean()),
                    longitude=float(map_df["display_longitude"].mean()),
                    zoom=10.8,
                    pitch=0,
                    bearing=0,
                )

                st.pydeck_chart(
                    pdk.Deck(
                        layers=layers,
                        initial_view_state=view,
                        tooltip={
                            "text": "{tooltip_label}",
                            "style": {"backgroundColor": "#063b4c", "color": "white"},
                        },
                    ),
                    use_container_width=True,
                    height=560,
                )

                if coastline_paths and display_coords is not None:
                    st.success(
                        "🗺️ Garis pantai Magepanda 2024 berhasil dimuat dari file lokal, dan "
                        "titik prediksi sintetis disebarkan mengikuti "
                        "garis pantai khusus untuk visualisasi. Koordinat asli dataset tetap tersimpan."
                    )
                else:
                    st.warning(
                        "⚠️ Garis pantai referensi tidak dapat dimuat. "
                        f"Peta tetap menampilkan titik dataset. Detail: {coastline_source}"
                    )

                st.markdown(
                    """
                    <div class="info-box">
                    <b>Legenda:</b><br>
                    🔴 Abrasi &nbsp;&nbsp; 🟡 Stabil &nbsp;&nbsp; 🟢 Akresi &nbsp;&nbsp; ◉ Referensi Magepanda<br>
                    <small>Garis biru-toska = garis pantai referensi Magepanda 2024. Ukuran titik = tingkat risiko prototype.</small><br>
                    <small><b>Penting:</b> garis pantai referensi bukan hasil survei OJOKNUSA. Pada visualisasi ini, koordinat yang mengikuti garis pantai adalah <b>koordinat visualisasi</b> untuk dataset sintetis; kolom latitude/longitude asli tidak diubah.</small>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.caption(
                    "Sumber geometri garis pantai: GarisPantai_Magepanda_2024.gpkg (dikonversi ke GeoJSON lokal). "
                    "Data prediksi OJOKNUSA tetap bersifat sintetis/purwarupa."
                )

                st.subheader("📊 Distribusi Kelas Prediksi")
                st.bar_chart(map_df["kelas_prediksi"].value_counts())

                st.subheader("⚠️ Distribusi Risiko Prototype")
                st.bar_chart(map_df["risiko_prototype"].value_counts())

                st.subheader("🔴 Titik Risiko Prototype Tinggi")
                high = map_df[map_df["risiko_prototype"] == "Tinggi"].copy()
                if len(high):
                    cols = [
                        "segment_id", "year", "latitude", "longitude",
                        "prediksi_change_rate_m_year", "kelas_prediksi",
                        "distance_to_road_m", "risiko_prototype",
                        "skor_risiko", "jarak_ke_magepanda_km"
                    ]
                    cols = [c for c in cols if c in high.columns]
                    st.dataframe(
                        high[cols].sort_values("skor_risiko", ascending=False),
                        use_container_width=True
                    )
                else:
                    st.info("Belum ada titik dengan risiko prototype tinggi pada tahun ini.")

                csv_peta = map_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download hasil analisis peta",
                    data=csv_peta,
                    file_name=f"ojoknusa_peta_risiko_{selected_year}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

                st.caption(
                    "⚠️ Kategori risiko merupakan aturan prototype dan belum "
                    "merupakan indeks risiko ilmiah tervalidasi."
                )


# =========================================================
# TENTANG OJOKNUSA
# =========================================================

elif st.session_state.menu == "Tentang OJOKNUSA":

    st.markdown(
        """
        <div class="home-hero">
            <h1>🌊 OJOKNUSA AI</h1>
            <p>
                Platform prototype berbasis Artificial Intelligence untuk analisis
                perubahan garis pantai dan analisis awal risiko abrasi terhadap
                infrastruktur pesisir.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(
            """
            <div class="about-card">
                <h3>🤖 Model AI</h3>
                <p>
                    Prototype menggunakan <b>Random Forest Regressor</b> untuk
                    memperkirakan laju perubahan garis pantai.
                </p>
                <p><b>Target:</b> <code>change_rate_m_year</code></p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="about-card">
                <h3>📊 Parameter Input</h3>
                <ul>
                    <li>Tahun</li>
                    <li>Latitude</li>
                    <li>Longitude</li>
                    <li>Tinggi gelombang</li>
                    <li>Kecepatan angin</li>
                    <li>Elevasi</li>
                    <li>Jarak terhadap jalan</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        """
        <div class="about-card">
            <h3>🔄 Alur Sistem</h3>
            <p>
                <b>Data → Preprocessing → Random Forest → Prediksi →
                Klasifikasi → Analisis Risiko → Visualisasi</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dashboard-note">
            ⚠️ <b>Status penelitian:</b> dataset pada tahap prototype merupakan
            <b>data sintetis/purwarupa</b>, bukan data observasi lapangan atau
            hasil penginderaan jauh aktual. Hasil aplikasi perlu divalidasi
            menggunakan data garis pantai aktual sebelum digunakan untuk
            pengambilan keputusan infrastruktur.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-box">
            ℹ️ <b>Catatan:</b> OJOKNUSA AI dirancang sebagai dashboard
            penelitian/prototype untuk membantu eksplorasi perubahan garis
            pantai, bukan sebagai pengganti survei lapangan.
        </div>
        """,
        unsafe_allow_html=True
    )
