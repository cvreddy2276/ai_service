# Breast DCE-MRI pCR Research Platform

> **Research prototype:** this repository does not currently contain a verified held-out evaluation report or a clinically validated model. The app estimates pathological complete response (pCR); it does not diagnose malignancy. Grad-CAM is an attribution visualization, not tumor segmentation or volumetry. Do not use outputs for diagnosis or treatment decisions.


A full-stack research pipeline for exploring pCR response estimation from 3D/4D DCE-MRI. It includes PyTorch/MONAI training and segmentation code, a FastAPI inference service, a Node.js API, and a React interface. It does not establish clinical performance, diagnose malignancy, or calculate tumor volume.

- **Web application**: React interface for scan uploads, saved research outputs, and audit views.
- **Backend API**: Node.js + Express API with authentication, role authorization, research summary export, and audit logging.
- **AI service**: FastAPI inference with canonical orientation, 1 mm resampling, matched normalization/cropping, pCR score, and Grad-CAM attribution.
- **Model code**: PyTorch/MONAI classifiers and 3D UNet segmentation experiments. The classification experiment has a patient-disjoint split workflow; its results have not been run or verified here.

---

## Complete Project Structure

```text
├── ai_service/             # FastAPI Deep Learning Microservice
│   ├── app/
│   │   ├── config.py       # Checkpoint paths and hardware device config
│   │   ├── predictor.py    # PyTorch 3D pCR inference + illustrative Grad-CAM
│   │   ├── preprocessing.py# Canonical NIfTI loading, 1mm resampling, z-score normalization
│   │   ├── schemas.py      # Pydantic request/response schemas
│   │   └── main.py         # FastAPI inference and research metrics endpoints
│   ├── Dockerfile          # AI microservice container
│   └── requirements.txt    # Python dependencies
├── client/                 # React.js Modern Healthcare Frontend (Vite + Tailwind CSS)
│   ├── src/
│   │   ├── components/     # Navbar, Sidebar, SliceViewer (Grad-CAM), StatCard, RiskBadge
│   │   ├── context/        # AuthContext (JWT, session, role switching)
│   │   ├── pages/          # Landing, Login, Register, Dashboard, Upload, Results, Patients, Analytics, AuditLogs
│   │   ├── services/       # Axios API client
│   │   ├── App.jsx         # Router & protected layouts
│   │   └── index.css       # Medical glassmorphic styling
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── nginx.conf          # Nginx production configuration
│   └── Dockerfile          # Multi-stage production container
├── server/                 # Node.js + Express Backend REST API
│   ├── src/
│   │   ├── config/         # MongoDB Mongoose & In-Memory fallback store
│   │   ├── controllers/    # auth, patient, prediction, report, analytics, audit controllers
│   │   ├── middleware/     # JWT protect, role authorization, audit logger, Multer NIfTI upload
│   │   ├── models/         # User, Patient, Prediction, Report, AuditLog schemas
│   │   ├── routes/         # Express REST routes
│   │   ├── services/       # AI service client, PDFKit medical report generator, DataStore
│   │   ├── app.js          # Express app middleware and error handlers
│   │   └── server.js       # Server entrypoint
│   ├── uploads/            # Uploaded NIfTI volumes
│   ├── reports/            # Generated research summary PDFs
│   ├── package.json
│   └── Dockerfile          # Express production container
├── checkpoints/            # Existing PyTorch weights; provenance/performance requires independent verification
├── data/                   # Clinical datasets, dummy cohorts, and NIfTI volumes
├── src/                    # Python deep learning modules (models, segmentation, preprocessing, utils)
├── docker-compose.yml      # Orchestrates client, server, ai_service, and mongodb
├── DEPLOYMENT.md           # Step-by-step Vercel, Render, Docker, MongoDB Atlas deployment guide
├── AGENTS.md               # Medical imaging coding standards & rules
└── README.md               # Project documentation
```

---

## Core Technologies

- **PyTorch (`torch`)**: Volumetric deep neural network foundations, backpropagation, and optimization.
- **MONAI (`monai`)**: Medical Open Network for AI for 3D medical data loaders, transforms, 3D UNet architectures, sliding-window inference, and metrics (Dice, HD95).
- **SimpleITK & NiBabel**: NIfTI I/O, physical space coordinate transformations, isotropic resampling, and N4 bias field correction.
- **Pure Deep Learning**: All predictive modeling is executed exclusively with PyTorch and MONAI neural architectures (no classical ML models like XGBoost, Random Forest, or SVM).

---

## Patient-Level pCR Experiment

Run a deterministic stratified patient-disjoint 70/15/15 train/validation/test experiment on the matched real cohort:

```powershell
.\.venv\Scripts\python.exe -m src.classification.experiment --data-dir BreastDCEDL_spy1 --metadata-csv BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv --epochs 30
```

The validation split selects the checkpoint. The test split is evaluated after training and the report records the exact patient IDs in `reports/experiments/pcr_patient_split/held_out_test_report.json`. This is one exploratory split; repeat it across seeds or use nested/grouped cross-validation, then seek external-cohort validation before making performance claims. The experiment does not automatically replace the service checkpoint.

### Patient-Grouped Cross-Validation

Run stratified five-fold patient-grouped outer evaluation with a separate inner validation set for checkpoint selection and early stopping:

```powershell
.\.venv\Scripts\python.exe -m src.classification.cross_validation --data-dir BreastDCEDL_spy1 --metadata-csv BreastDCEDL_spy1/BreastDCEDL_spy1_metadata.csv --n-splits 5 --max-epochs 20 --patience 5
```

The output includes fold metrics, pooled out-of-fold predictions, and bootstrap intervals under `reports/evaluation/pcr_grouped_5fold/`. This is internal validation on the available Duke cohort. It does not replace an independent external-cohort evaluation, and the fold checkpoints are not deployment checkpoints.

## Quick Start & Verification

### 1. Verify Deep Learning Environment
Verifies PyTorch, MONAI 3D models, SimpleITK, and dependencies:
```powershell
.\.venv\Scripts\python.exe -m src.utils.verify_setup
```

### 2. Run 3D Deep Learning Classification Verification
Runs a synthetic dry-run check of the classifier code. This is not real-cohort training or an accuracy evaluation:
```powershell
.\.venv\Scripts\python.exe -m src.utils.test_classification
```

### 3. Run 3D Segmentation Verification
Executes a synthetic MONAI 3D Residual UNet dry run. This is not held-out segmentation validation:
```powershell
.\.venv\Scripts\python.exe -m src.utils.test_segmentation
```

### 4. Run Reinforcement Learning Co-Training Verification
Validates the Deep Q-Network (DQN) agent, action selection, Bellman updates, and 2-epoch dry runs for both segmentation and classification:
```powershell
.\.venv\Scripts\python.exe -m src.utils.test_rl_training
```

### 5. Run Preprocessing Pipeline Verification
Tests multi-temporal DCE loading, isotropic resampling, and peak subtraction ($I_{\text{sub}} = \max(T_1 - T_0, 0)$):
```powershell
.\.venv\Scripts\python.exe -m src.utils.test_preprocessing
```

### 6. Run 3D Radiomics Feature Extraction Verification
Extracts 61 IBSI-compliant 3D morphological, first-order, and texture features:
```powershell
.\.venv\Scripts\python.exe -m src.utils.test_radiomics
```

### 7. Launch Interactive Jupyter Notebook Demo
Launches Jupyter Lab to run [`notebooks/breast_dce_dl_pipeline.ipynb`](notebooks/breast_dce_dl_pipeline.ipynb):
```powershell
.\.venv\Scripts\jupyter-lab.exe
```

---

## Experimental Reinforcement Learning Utilities

The DQN utilities are exploratory code and are not part of the pCR training/evaluation workflow described above. Their dry-run output should not be interpreted as model accuracy or a validated improvement.

The framework incorporates an active reinforcement learning controller ([`src/rl/patch_sampling_agent.py`](src/rl/patch_sampling_agent.py)) that dynamically directs the training data stream:

- **State Vector ($S_t \in \mathbb{R}^5$)**:
  $$\mathbf{s}_t = [\tanh(\mathcal{L}), \text{Metric}, \Delta \mathcal{L}, \text{Progress}, \text{PolicyFactor}]$$
- **Action Policies ($A_t \in \{0, 1, 2, 3, 4\}$)**:
  - `0: tumor_center_patch` — Centers patches on the contrast-enhancing tumor core (`pos=1.0, neg=0.05`).
  - `1: lesion_boundary_patch` — Focuses on the tumor-parenchyma transition margin (`pos=0.6, neg=0.4`).
  - `2: background_patch` — Hard negative mining of healthy parenchyma and chest wall (`pos=0.05, neg=0.95`).
  - `3: increase_augmentation` — Increases 3D rotation, flip, and noise perturbation to prevent overfitting.
  - `4: decrease_augmentation` — Reduces augmentations to stabilize convergence during difficult epochs.
- **Reward Signal**:
  $$R_t = \Delta \text{Metric} - 0.1 \times \max(\Delta \mathcal{L}, 0)$$
- **Optimization**: Transitions $(\mathbf{s}_t, a_t, r_t, \mathbf{s}_{t+1}, \text{done})$ are stored in replay memory and optimized via Smooth L1 Bellman loss.

---

## Python API Usage

### RL-Guided 3D UNet Segmentation Training
```python
from src.rl import RLSegmentationTrainer

trainer = RLSegmentationTrainer(
    train_data=train_samples,
    val_data=val_samples,
    spatial_size=(96, 96, 32),
    batch_size=2,
    lr=1e-3,
)
results = trainer.fit(max_epochs=20)
print("Best Validation Dice:", results["best_val_dice"])
```

### RL-Guided 3D Treatment Response Classification
```python
from src.rl import RLClassificationTrainer

trainer = RLClassificationTrainer(
    train_samples=train_samples,
    val_samples=val_samples,
    backbone="simple_cnn3d",  # or 'resnet18', 'densenet121'
    batch_size=2,
    lr=1e-4,
)
results = trainer.fit(epochs=20)
print("Best Validation ROC-AUC:", results["best_val_auc"])
```

---

## Diagnostic Figures & Outputs

Available artifacts include historical dry-run outputs and experiment results when the patient-level workflow is run:
- Checkpoints: `checkpoints/` (`best_monai_3dunet.pt`, `best_3d_classifier.pt`, `best_rl_monai_3dunet.pt`, etc.)
- Diagnostic Figures: `reports/figures/` (`dl_roc_curve.png`, `dl_confusion_matrix.png`, `dl_training_curves.png`, `rl_training_diagnostic.png`)
- Historical dry-run report: `reports/classification_report.json` (four-case verification output; not a performance estimate).
- New patient-disjoint experiment reports: `reports/experiments/pcr_patient_split/held_out_test_report.json`.
