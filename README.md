# CHAPATI ANALYZER™

> **"Because apparently, even a chapati needs a performance review."**  
> *"We took something that should be judged by someone's grandmother and turned it into a computer-vision problem."*

---

## 1. Project Overview & Motivation

**Chapati Analyzer™** is an intentionally over-engineered computer-vision Progressive Web App (PWA) that extracts measurable mathematical and optical properties of a chapati from a photograph or phone camera capture to answer:

> **"How mathematically good is this chapati?"**

Instead of human subjectivity or black-box neural hallucinations, this system employs **classical digital image processing, contour calculus, radial signal analysis, and deterministic scoring**.

---

## 2. Key Features

- 📱 **True Progressive Web App (PWA)**: Installable on Android, iOS, tablet, and desktop with standalone display and offline shell caching.
- 📷 **First-Class Mobile Camera Support**: Live viewfinder reticle, rear camera priority (`facingMode: "environment"`), camera flip, snapshot confirmation, and drag-and-drop file upload fallback.
- 📐 **Rigorous Geometry**:
  - Exact contour area (Green's theorem) & perimeter
  - Circularity index ($4\pi A / P^2$)
  - Aspect ratio & equivalent diameter
  - Taubin/Kåsa algebraic circle fitting ("The circle your chapati was trying to be")
  - Mass centroid vs. fitted circle center deviation
- 🔄 **360° Radial & Symmetry Analysis**:
  - Sampled radial profile $r(\theta)$ for $\theta \in [0^\circ, 360^\circ)$
  - Radius stability score (coefficient of variation $CV = \sigma / \mu$)
  - $180^\circ$ and $90^\circ$ rotational symmetry
  - Edge irregularity & high-frequency smoothness filtering
- 🔥 **Concentric Browning & Toasting Analysis**:
  - Maillard reaction patch segmentation
  - 5 concentric radial zones (Center, Inner, Middle, Outer, Edge)
  - Burn ratio and humorous pattern classifications (*Balanced Browning, Center Burner, Edge Burner, etc.*)
- 🔬 **Visible Texture Analysis**:
  - Surface Laplacian variance and local contrast
- 🏆 **Chapati Perfection Index™ (0–100)**:
  - 100% deterministic weighted scoring (zero fake or random numbers)
  - 6 certified verdict tiers with humorous quotes
- 💾 **Persistence & Exports**:
  - SQLite database (`data/chapati.db`) with sequential specimen IDs (`CHAPATI-0001`)
  - Full analysis JSON export
  - Registry CSV export
  - Certified high-resolution laboratory report card download
- 🧬 **Stage 2 Readiness**:
  - Internal 360-bin normalized shape vectors preserved for the planned **Chapati Shape Genome™** and evolutionary trees.

---

## 3. Mathematical Formulas & Pipeline

### Computer Vision Pipeline
```text
INPUT IMAGE
    ↓
PREPROCESSING (Resize, Bilateral Filter, HSV & CIE Lab conversions)
    ↓
SEGMENTATION (Multi-strategy Otsu, dough color thresholding, candidate scoring)
    ↓
GEOMETRIC ANALYSIS (Area, perimeter, circularity, Taubin circle fit)
    ↓
RADIAL ANALYSIS (360° ray profile r(θ), stability, 180° symmetry, edge filter)
    ↓
BROWNING & TEXTURE (5-zone Maillard quantification, Laplacian variance)
    ↓
PERFECTION INDEX (Deterministic weighted synthesis)
    ↓
VERDICT & REPORT GENERATION (Visual overlays, database storage, exports)
```

### Core Formulations
- **Circularity**:
  $$\text{Circularity} = \frac{4\pi A}{P^2}$$
- **Equivalent Diameter**:
  $$D = 2\sqrt{\frac{A}{\pi}}$$
- **Radius Stability**:
  $$CV = \frac{\sigma_r}{\mu_r}, \quad \text{Stability} = \max(0, 100 \times (1 - 2.5 \times CV))$$
- **Rotational Symmetry (180°)**:
  $$\text{Diff}_{180} = \frac{1}{360} \sum_{\theta=0}^{359} \frac{|r(\theta) - r((\theta+180)\bmod 360)|}{\mu_r}$$
- **Burn Ratio**:
  $$\text{Burn Ratio} = \frac{\text{Browned Area}}{\text{Chapati Area}}$$

### Perfection Index Weights
| Feature | Weight |
|---|---:|
| Circularity | 30% |
| Radius Stability | 20% |
| Rotational Symmetry | 15% |
| Edge Smoothness | 15% |
| Center Accuracy | 10% |
| Browning Control | 5% |
| Visual Texture | 5% |
| **Total** | **100%** |

### Verdict Hierarchy
- **95–100**: `GEOMETRICALLY ENLIGHTENED` — *"This chapati has achieved a level of circularity that may concern mathematicians."*
- **90–95**: `ALMOST CIRCULAR` — *"Grandma would probably approve."*
- **80–90**: `ACCEPTABLE CHAPATI` — *"Slightly geographically confused, but operational."*
- **70–80**: `QUESTIONABLE GEOMETRY` — *"The circle was merely a suggestion."*
- **60–70**: `SHAPE INCIDENT` — *"Something happened here."*
- **< 60**: `PLEASE CONSULT A CHAPATI ENGINEER` — *"Immediate geometric intervention recommended."*

---

## 4. Technology Stack

- **Backend**: Python 3.12, Flask, OpenCV (`opencv-python`), NumPy, SciPy, Matplotlib, Pillow, SQLite3
- **Frontend**: HTML5, CSS3, ES6 JavaScript Modules (zero framework bloat)
- **PWA**: Web App Manifest, Service Worker (`sw.js`), Cache API, MediaDevices Camera API

---

## 5. Installation & Running

### 1. Prerequisites
Python 3.10+ installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python app.py
```
Open your browser at:
```text
http://localhost:5000
```
*(To test on a mobile phone on the same Wi-Fi, open `http://<YOUR_COMPUTER_IP>:5000`)*

---

## 6. Running Tests

Run the comprehensive unit test suite using synthetic mathematical shapes:
```bash
python -m unittest discover -s tests -v
```

---

## 7. Limitations & Scientific Disclaimer

1. **Uncalibrated Dimensions**: Pixel dimensions (area, diameter) are reported in pixels unless a standardized physical calibration marker is used.
2. **Sensory Disclaimer**: This application analyzes visible optical and geometric features from images. It does **not** determine physical chewiness, softness, internal puffiness, nutritional value, freshness, taste, or food safety.
