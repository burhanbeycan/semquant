from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st
import imageio.v3 as iio

from examples.generate_synthetic import synthetic_fibers, synthetic_particles, synthetic_pores
from semquant.defaults import load_default_config
from semquant.pipeline import analyze


def zip_dir(folder: Path) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))
    return buf.getvalue()


st.set_page_config(page_title="SEMQuant", layout="wide")
st.title("SEMQuant — SEM Image Quantification (ImageJ‑like)")

cfg_default = load_default_config()

with st.sidebar:
    st.header("Analysis settings")
    mode = st.selectbox("Mode", ["pores", "particles", "fibers"], index=0)

    st.subheader("Calibration")
    pixel_size = st.number_input("Pixel size (µm/px). Leave 0 to use scale bar detection.", min_value=0.0, value=0.0)
    scale_bar_length = st.number_input("Scale bar real length", min_value=0.0, value=1.0)
    scale_bar_unit = st.selectbox("Scale bar unit", ["nm", "um", "mm"], index=1)

    st.subheader("Segmentation")
    thr_method = st.selectbox("Threshold method", ["otsu", "yen", "sauvola"], index=0)
    invert = st.checkbox("Invert mask", value=False)
    gaussian_sigma = st.slider("Gaussian sigma", 0.0, 5.0, 1.0, 0.1)
    exclude_bottom = st.slider("Exclude bottom fraction (remove scale bar/legend)", 0.0, 0.5, 0.12, 0.01)
    min_area_px = st.number_input("Min object area (px)", min_value=0, value=80, step=10)

    make_pdf = st.checkbox("Make PDF report", value=True)

uploaded = st.file_uploader("Upload an SEM image (png/jpg/tif)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"])
with st.expander("No SEM file yet? Try a built-in demo image"):
    st.caption("Generate a synthetic SEM-like image and run the same pipeline with one click.")
    use_demo = st.button("Use demo image")

if use_demo:
    demo_builders = {
        "pores": synthetic_pores,
        "particles": synthetic_particles,
        "fibers": synthetic_fibers,
    }
    demo_img = (demo_builders[mode](seed=42) * 255).astype("uint8")
    demo_bytes = iio.imwrite("<bytes>", demo_img, extension=".png")
    uploaded = io.BytesIO(demo_bytes)
    uploaded.name = f"demo_{mode}.png"
    st.info(f"Using generated demo image: {uploaded.name}")

if uploaded is None:
    st.info("Upload an SEM image to begin.")
    st.stop()

tmpdir = Path(tempfile.mkdtemp(prefix="semquant_"))
img_path = tmpdir / uploaded.name
img_path.write_bytes(uploaded.getvalue())

cfg = cfg_default
cfg["analysis"]["mode"] = mode
cfg["analysis"]["threshold_method"] = thr_method
cfg["analysis"]["invert"] = bool(invert)
cfg["analysis"]["gaussian_sigma"] = float(gaussian_sigma)
cfg["analysis"]["exclude_bottom_fraction"] = float(exclude_bottom)
cfg["cleanup"]["min_object_area_px"] = int(min_area_px)
cfg["output"]["make_pdf_report"] = bool(make_pdf)

out_dir = tmpdir / "out"
out_dir.mkdir(parents=True, exist_ok=True)

run = st.button("Run analysis", type="primary")
if not run:
    st.stop()

with st.spinner("Running SEMQuant..."):
    summary = analyze(
        img_path,
        out_dir=out_dir,
        config=cfg,
        mode=mode,
        pixel_size_um=(pixel_size if pixel_size > 0 else None),
        scale_bar_length=(scale_bar_length if pixel_size <= 0 else None),
        scale_bar_unit=scale_bar_unit,
    )

st.success("Done!")

st.subheader("Summary")
st.dataframe(pd.DataFrame([summary]).T.rename(columns={0: "value"}), use_container_width=True)

st.subheader("Key figures")
figs = [
    out_dir / "figures" / "01_original.png",
    out_dir / "figures" / "03_overlay.png",
    out_dir / "figures" / "hist_diameter.png",
]
cols = st.columns(3)
for c, p in zip(cols, figs):
    if p.exists():
        c.image(str(p), use_column_width=True)

st.subheader("Download")
zip_bytes = zip_dir(out_dir)
st.download_button(
    label="Download results.zip",
    data=zip_bytes,
    file_name="results.zip",
    mime="application/zip",
)

if st.button("Delete temporary files"):
    shutil.rmtree(tmpdir, ignore_errors=True)
    st.toast("Deleted.")
