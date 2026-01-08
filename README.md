# Structured Synthetic Augmentation as Data Regularization for Spatio-Temporal Anomaly Classification (fMRI + Video)

This repository contains the code used to reproduce the main experimental pipeline and the **cross-domain generalization stress-tests** for binary anomaly classification in spatio-temporal data.

The core idea is **structured synthetic augmentation** as a form of *data-level regularization*: generating semantically plausible perturbations in a constrained region/structure (e.g., ROI mask in fMRI, or local geometry in feature space for video) to improve stability under class imbalance and distribution shifts.



---

## Highlights

- **Cross-domain evaluation (stress-test):** train on one domain, evaluate on fully independent external domains the model never saw.
- **Ablation:** *structured* vs *unstructured (noise)* augmentation under the same architecture and training protocol.
- **Baselines:** reconstruction-based **autoencoder** baseline evaluated under the same cross-domain protocol.
- **Subject-level protocol for fMRI:** strict separation by subject; reported metrics are computed at **subject level**.
- **Threshold calibration:** for thresholded metrics (F1/ACC), the threshold is selected on a calibration split using **Youden’s J statistic**.
- **Reproducibility controls:** multiple fixed random seeds (default: `42, 43, 44`) with aggregation `mean ± std`.
- **Artifacts saved:** CSV tables + plots (ROC, t-SNE, augmentation visualization) produced by the notebook.

---

## Repository contents

- `MMRO_acticle_code.ipynb` — end-to-end notebook (download → preprocess → train → evaluate → export artifacts)


---

## Environment

### Option A - Google Colab (recommended for reviewers)
1. Upload/open `MMRO_acticle_code.ipynb` in Colab.
2. Ensure GPU runtime is enabled (`Runtime → Change runtime type → GPU`).
3. Run cells top-to-bottom.

The notebook is designed to **download datasets from scratch** and produce final tables/plots automatically.

### Option B - Local (Linux/macOS)
**Prerequisites**
- Python 3.10+ (3.11/3.12 also works in many setups)
- CUDA-enabled GPU recommended (but CPU may work for limited smoke tests)
- `awscli` installed (for OpenNeuro public S3 access)
- Typical Python packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `nibabel`, `tqdm`, `matplotlib`

Example setup:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Minimal dependencies (adjust to your needs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pandas scikit-learn nibabel tqdm matplotlib
pip install awscli
```
Then run:
```bash
jupyter notebook MMRO_acticle_code.ipynb
```
## Data
### fMRI (OpenNeuro)

The notebook downloads public OpenNeuro datasets via AWS S3 (no-sign-request).
The exact dataset IDs and their roles are configurable in the notebook; the default rebuttal stress-test setup uses:

Source-domain normal (train/val): ds003688
External normal (never seen in training): ds001769 (Forrest Gump; phantom subject excluded)
External anomaly domain 1: ds003999 (stroke)
External anomaly domain 2 (optional): ds004884 (clinical)
Control with matched experimental task (rest): ds001168 (normal rest) vs ds003999 (rest)

### Protocol notes

Unit of observation: subject
One BOLD file per subject is selected (to avoid pseudo-replication).
Evaluation pools are balanced per domain where applicable (e.g., n_per_ds).
Thresholded metrics use a calibration split; ranking metrics use full score distributions.
The notebook stores downloads under a local directory such as:
```bash
/content/openneuro_s3/   (Colab)
./openneuro_s3/          (local)
```

## Note on Video Experiments

The video anomaly detection experiments referenced in this repository are **not intended as a full empirical study of the video domain** and are included only to illustrate the applicability of the proposed structured augmentation principle beyond medical data.

A comprehensive experimental analysis of video anomaly detection, including architectural details, large-scale benchmarking, and extensive ablation studies, is presented in a separate peer-reviewed publication:

**A. K. Zvereva, M. S. Kaprielova, A. V. Grabovoy.**  
*AnomLite: Efficient Binary and Multiclass Video Anomaly Detection.*  
Results in Engineering, 25 (2025), 104162.  
DOI: https://doi.org/10.1016/j.rineng.2025.104162

Readers interested in video-specific methodology and results are referred to the above work.

