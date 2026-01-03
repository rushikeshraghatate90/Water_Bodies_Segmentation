# water_analyzer_app.py
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import io
import os
from typing import Tuple, Any, Optional

# ----------------- Page Configuration -----------------
st.set_page_config(
    page_title="🛰️ Satellite Water Analyzer",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- COLORFUL & MODERN CSS -----------------
st.markdown("""
<style>
    /* Main App background and font */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', sans-serif;
    }
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Sidebar styling */
    .st-emotion-cache-16txtl3 {
        background-color: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0,0,0,0.1);
    }
    /* Title styling */
    h1 {
        background: -webkit-linear-gradient(45deg, #09477e, #00d4ff 80%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 900;
    }
    /* Metric styling */
    .st-emotion-cache-1g6gooi {
        background-color: #FFFFFF;
        border-left: 7px solid #00d4ff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
    }
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        color: white;
        font-weight: bold;
        border: none;
        background-image: linear-gradient(45deg, #00c6ff 0%, #0072ff 100%);
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,114,255,0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,114,255,0.4);
    }
    /* Image and Plot containers */
    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
		gap: 24px;
	}
	.stTabs [data-baseweb="tab"] {
		height: 50px;
        white-space: pre-wrap;
		background-color: transparent;
		border-radius: 4px 4px 0px 0px;
		gap: 1px;
		padding-top: 10px;
		padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-image: linear-gradient(45deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Session State Initialization -----------------
if 'processed' not in st.session_state:
    st.session_state.processed = False
    st.session_state.analysis_data = {}

# ----------------- Core Processing Functions -----------------
def load_and_resize(image_file: Any) -> Optional[Image.Image]:
    """Loads an image, handles errors, and resizes it to 256x256."""
    try:
        img = Image.open(image_file).convert("RGB")
        return img.resize((256, 256))
    except (UnidentifiedImageError, Exception) as e:
        st.error(f"Error loading image: {e}. Please upload a valid image file.")
        return None

def create_overlay(image_np: np.ndarray, mask_np: np.ndarray, color: tuple, alpha: float = 0.5) -> np.ndarray:
    """Creates a colorful, transparent overlay of the mask on the image."""
    overlay = image_np.copy()
    colored_mask = np.zeros_like(overlay)
    colored_mask[mask_np > 128] = color
    return cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)

def get_boundary_edges(mask_np: np.ndarray) -> np.ndarray:
    """Finds the boundary of the mask using Canny edge detection."""
    return cv2.Canny(mask_np, 100, 200)

def analyze_water_bodies(mask_np: np.ndarray) -> Tuple[int, int]:
    """Counts distinct water bodies and finds the area of the largest one."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_np, 4, cv2.CV_32S)
    if num_labels <= 1: return 0, 0
    # Areas are in the last column of stats. Index 0 is the background.
    areas = stats[1:, cv2.CC_STAT_AREA]
    return num_labels - 1, int(np.max(areas))

def create_pie_chart(water_pixels: int, land_pixels: int) -> Optional[plt.Figure]:
    """Generates a styled Matplotlib donut chart."""
    if water_pixels + land_pixels == 0: return None
    labels = 'Water', 'Land'
    sizes = [water_pixels, land_pixels]
    colors = ['#0072ff', '#9CCC65']
    explode = (0.1, 0)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=120, pctdistance=0.85,
           textprops={'color': "white", 'weight': 'bold', 'fontsize': 12})
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    ax.axis('equal')
    fig.patch.set_alpha(0.0)
    return fig

def process_images(original_img_file: Any, mask_img_file: Any, overlay_color: str):
    """Main function to run all analyses and store results in session_state."""
    original_img = load_and_resize(original_img_file)
    mask_img = load_and_resize(mask_img_file)

    if original_img is None or mask_img is None: return

    original_np = np.array(original_img)
    mask_gray_np = np.array(mask_img.convert("L"))

    water_pixels = np.sum(mask_gray_np > 128)
    total_pixels = mask_gray_np.size
    land_pixels = total_pixels - water_pixels
    water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    num_bodies, largest_body_area = analyze_water_bodies(mask_gray_np)
    
    overlay_color_rgb = tuple(int(overlay_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
    overlay_img_np = create_overlay(original_np, mask_gray_np, overlay_color_rgb)
    edges_np = get_boundary_edges(mask_gray_np)
    edge_overlay_np = original_np.copy()
    edge_overlay_np[edges_np > 0] = [255, 0, 80]

    st.session_state.processed = True
    st.session_state.original_img = original_img
    st.session_state.mask_img = mask_img
    st.session_state.overlay_img = Image.fromarray(overlay_img_np)
    st.session_state.edge_img = Image.fromarray(edge_overlay_np)
    st.session_state.fig = create_pie_chart(water_pixels, land_pixels)
    st.session_state.analysis_data = {
        "Image": [original_img_file.name],
        "Mask": [mask_img_file.name],
        "Water_Pixels": [water_pixels],
        "Water_Percentage": [round(water_percentage, 2)],
        "Distinct_Water_Bodies": [num_bodies],
        "Largest_Body_Area_px": [largest_body_area],
    }

def get_image_download_bytes(img: Image.Image) -> bytes:
    """Converts a PIL Image to bytes for downloading."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ----------------- UI Layout: Sidebar -----------------
with st.sidebar:
    st.header("⚙️ Controls")
    
    uploaded_image = st.file_uploader("1. Upload Original Image", type=["jpg", "png", "jpeg"])
    uploaded_mask = st.file_uploader("2. Upload Mask Image", type=["jpg", "png", "jpeg"])

    st.color_picker("🎨 Select Overlay Color", "#0072ff", key="overlay_color")
    st.markdown("---")

    if st.button("🚀 Process Images", use_container_width=True):
        if uploaded_image and uploaded_mask:
            with st.spinner('Analyzing...'):
                process_images(uploaded_image, uploaded_mask, st.session_state.overlay_color)
        else:
            st.warning("Please upload both an original image and a mask file.")

    if st.button("Reset", use_container_width=True):
        st.session_state.processed = False
        st.rerun()

# ----------------- UI Layout: Main Page -----------------
st.title("Satellite Water Analyzer")

if not st.session_state.processed:
    st.info("👋 Welcome! Please upload an original satellite image and its corresponding mask in the sidebar to begin.")
    st.image("https://i.imgur.com/vWaG6A1.jpeg", caption="Analyze water bodies from satellite imagery.")
else:
    data = st.session_state.analysis_data
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🖼️ Visuals", "💾 Data Export"])

    with tab1:
        st.subheader("Key Metrics")
        cols = st.columns(4)
        cols[0].metric("Water Coverage", f"{data['Water_Percentage'][0]:.2f}%")
        cols[1].metric("Total Water Area", f"{data['Water_Pixels'][0]:,} px")
        cols[2].metric("Distinct Bodies", f"{data['Distinct_Water_Bodies'][0]}")
        cols[3].metric("Largest Body", f"{data['Largest_Body_Area_px'][0]:,} px")
        st.subheader("Water vs. Land Distribution")
        if st.session_state.fig:
            st.pyplot(st.session_state.fig, use_container_width=True)

    with tab2:
        st.subheader("Image Analysis")
        cols = st.columns(4)
        cols[0].image(st.session_state.original_img, caption="Original Image", use_container_width=True)
        cols[1].image(st.session_state.mask_img, caption="Segmentation Mask", use_container_width=True)
        cols[2].image(st.session_state.overlay_img, caption="Analysis Overlay", use_container_width=True)
        cols[3].image(st.session_state.edge_img, caption="Water Boundary", use_container_width=True)
            
        st.markdown("---")
        st.subheader("Download Results")
        d_cols = st.columns(4)
        d_cols[0].download_button("Download Original", get_image_download_bytes(st.session_state.original_img), "original.png", "image/png", use_container_width=True)
        d_cols[1].download_button("Download Mask", get_image_download_bytes(st.session_state.mask_img), "mask.png", "image/png", use_container_width=True)
        d_cols[2].download_button("Download Overlay", get_image_download_bytes(st.session_state.overlay_img), "overlay.png", "image/png", use_container_width=True)
        d_cols[3].download_button("Download Boundary", get_image_download_bytes(st.session_state.edge_img), "boundary.png", "image/png", use_container_width=True)

    with tab3:
        st.subheader("Analysis Data")
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        csv_file = "water_analysis_log.csv"
        df_new = pd.DataFrame(data)
        
        if st.button("Append to Analysis Log (CSV)", use_container_width=True):
            try:
                if os.path.exists(csv_file):
                    df_existing = pd.read_csv(csv_file)
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_final = df_new
                df_final.to_csv(csv_file, index=False)
                st.success(f"✅ Analysis appended to `{csv_file}`")
            except Exception as e:
                st.error(f"Failed to save CSV: {e}")

        with st.expander("📂 Manage Analysis Log"):
            if os.path.exists(csv_file):
                st.write(f"**Current content of `{csv_file}`:**")
                st.dataframe(pd.read_csv(csv_file), use_container_width=True)
                if st.button("Clear Log File", type="primary"):
                    os.remove(csv_file)
                    st.warning(f"🗑️ `{csv_file}` has been deleted.")
                    st.rerun()
            else:
                st.info("Log file does not exist yet. Save an analysis to create it.")
