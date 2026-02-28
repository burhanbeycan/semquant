# SEMQuant — Automated SEM Image Quantification (ImageJ‑like workflow)

SEMQuant is a lightweight, reproducible pipeline to extract common quantitative descriptors from **Scanning Electron Microscopy (SEM)** images with **one command** (or a simple upload UI):

- **Porosity (%)** and **pore size distribution**
- **Particle size distribution** (equivalent circular diameter, Feret-like axes, circularity, etc.)
- **Fiber diameter distribution** (skeleton + distance transform; useful for electrospun mats)
- Publication‑ready outputs: **annotated images**, **histograms/CDF plots**, **CSV tables**, and an optional **PDF report**

The goal is to feel familiar to ImageJ users (threshold → mask → label → measure), while being:
- scriptable (batch processing),
- transparent (YAML config + saved parameters),
- and easy to share (outputs folder is self‑contained and zip‑ready).

---

## What you get (outputs)

For each analyzed image, SEMQuant creates an output folder like:

```
out/
  figures/
    01_original.png
    02_mask.png
    03_overlay.png
    04_labeled.png
    hist_diameter.png
    cdf_diameter.png
  tables/
    measurements.csv
    summary.csv
  masks/
    mask.png
  report/
    report.md
    report.pdf   # optional
  run_info/
    config_used.yaml
    versions.json
  results.zip    # optional (created by the Streamlit app)
```

---

## Installation

### Option A — pip install (editable for development)

```bash
git clone https://github.com/<your-org-or-user>/SEMQuant.git
cd SEMQuant
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -U pip
pip install -e .[dev,ui]
```

### Option B — plain requirements

```bash
pip install -r requirements.txt
```

---

## Quickstart (CLI)

### 1) Porosity + pore sizes

```bash
semquant analyze path/to/image.tif \
  --mode pores \
  --out out_pores \
  --scale-bar-length 1.0 --scale-bar-unit um
```

### 2) Particle size distribution

```bash
semquant analyze path/to/image.png \
  --mode particles \
  --out out_particles \
  --pixel-size 0.0125   # µm / pixel
```

### 3) Fiber diameter distribution (electrospun mats)

```bash
semquant analyze path/to/fibers.jpg \
  --mode fibers \
  --out out_fibers \
  --pixel-size 0.004
```

### Batch processing a folder

```bash
semquant batch ./images --mode pores --out ./batch_out --pixel-size 0.01
```

---

## Quickstart (Upload UI)

If you prefer an “upload → analyze → download zip” workflow:

```bash
streamlit run app.py
```

The app lets you:
- upload an SEM image,
- pick analysis mode (pores / particles / fibers),
- enter calibration (pixel size or scale bar length),
- run the pipeline,
- and download a ready‑to‑share **results.zip**.

---

## Calibration (pixel size / scale bar)

Quantitative sizes require calibration. SEMQuant supports two approaches:

1) **Direct pixel size** (recommended if you know it):
- `--pixel-size 0.01` means **0.01 µm per pixel**.

2) **Scale bar detection** (semi‑automatic):
- SEMQuant can detect a long horizontal scale bar near the bottom of the image and measure its pixel length.
- You still provide the real-world scale bar length (because label OCR is not reliable across SEM exporters).

Example:
```bash
semquant analyze img.tif --mode pores --out out \
  --scale-bar-length 5 --scale-bar-unit um
```

If scale bar detection fails, SEMQuant will still run but will mark the run as **uncalibrated** (sizes stay in pixels).

---

## How the measurements are computed

### Porosity (pores mode)
- Threshold → pore mask (dark or bright pores configurable)
- Clean with morphology (remove small objects)
- **Porosity (%)** = pore area / ROI area × 100
- Pore size = connected component **equivalent diameter** (and optional axes)

### Particles (particles mode)
- Threshold → particle mask
- Label connected components
- Measures include: area, perimeter, equivalent diameter, major/minor axis, aspect ratio, circularity, solidity, orientation

### Fibers (fibers mode)
- Threshold → fiber mask
- Skeletonize fibers
- Distance transform of mask
- Fiber diameter at each skeleton pixel = 2 × distance-to-boundary
- Outputs a diameter distribution and summary stats

---

## Configuration

All parameters can be set via CLI flags **or** a YAML file.

- If you omit `--config`, SEMQuant uses a built‑in default config.

```bash
semquant analyze img.tif --config configs/default.yaml --out out
```

See `configs/default.yaml` for the full set of knobs.

---

## Limitations (practical SEM notes)
- Threshold-based segmentation is sensitive to contrast, charging, and detector settings.
- For **multi-phase polymer composites** or **uneven illumination**, you may want to:
  - increase smoothing (`gaussian_sigma`),
  - switch threshold method (Otsu ↔ Yen ↔ Sauvola),
  - or define an ROI crop to exclude legends/scale bars.

SEMQuant saves the **exact parameters** used for each run to support traceable, publication-grade reporting.

---

## Citing
If you use SEMQuant in academic work, please cite the repo and also acknowledge the underlying ecosystem:
- scikit-image, NumPy, SciPy, matplotlib, pandas.

A `CITATION.cff` file is included.

---

## License
MIT — see `LICENSE`.
