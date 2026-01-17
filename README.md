# Lane Detection and Steering Angle Estimation (U-Net + Kalman Filter)

Vision-based lane segmentation and steering angle estimation using a **custom U-Net (trained from scratch)** and **Kalman filter smoothing**.  
Designed for an academic autonomous vehicle perception → estimation → control pipeline.

---

## Key Features
- **Binary lane segmentation** (lane vs background) using a lightweight U-Net
- **Steering angle estimation** from lane boundary curvature (2nd-order polynomial fit)
- **Kalman filter (2D state: angle + angular rate)** for temporal smoothing
- Supports **stereo calibration rectification** (left/right alignment)  
  > Lane mask is computed from the **rectified LEFT** frame (right frame is for alignment/depth extension)

---

## Learning Outcomes Coverage (LO1–LO3)
- **LO1 (Comprehension):** explains core CV methods (segmentation, camera calibration, stereo geometry) and filtering (Kalman)
- **LO2 (Application):** implements a working lane segmentation + steering estimation pipeline
- **LO3 (Communication/Teamwork):** clear documentation, modular code, results + evaluation metrics

---

## Project Pipeline (High Level)

Stereo Camera → Rectification → (Left Frame) U-Net Segmentation → Binary Mask  
→ Boundary Extraction → Polynomial Curve Fit → Raw Angle  
→ Kalman Filter → Smoothed Steering Angle → (Optional) Voltage Mapping

---

## Folder Structure (Recommended)

Lane-Detection-and-Steering-Angle-Estimation/
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── src/
│   ├── train_unet.py
│   ├── evaluate_metrics.py
│   ├── inference_single_image.py
│   ├── stereo_rectify_and_steer.py
│   └── utils/
│       ├── data_loader.py
│       ├── unet_model.py
│       ├── geometry.py
│       └── kalman.py
│
├── notebooks/
│   ├── comvis_training.ipynb
│   └── sfv_stereo_steering.ipynb
│
├── models/
│   └── lane_model.h5              # trained model (optional to upload, see notes)
│
├── calibration/
│   └── stereo_calibration.npz     # stereo parameters (K, D, R, T)
│
├── data/                          # (DO NOT UPLOAD FULL DATASET)
│   └── README_DATA.md             # instructions to place dataset locally
│
└── results/
    ├── loss_accuracy.png
    ├── prediction_grid.png
    ├── confusion_matrix.png
    └── confusion_matrix_norm.png

> NOTE: Put large datasets in `data/` but keep them **ignored** by git.

---

## Dataset
- Source: **Roboflow lane segmentation dataset** (image + mask pairs)
- Input: RGB road images
- Ground truth: binary masks
  - `1` = lane/drivable region  
  - `0` = background

---

## Preprocessing
- Resize images and masks to **256×256**
  - makes training faster + consistent input size for CNN
- Normalize image pixels to **[0, 1]**
- Convert masks to **binary**
- Build `tf.data` pipeline: shuffle → batch → prefetch

---

## Model Architecture (Custom U-Net, Trained From Scratch)
This project uses a **simplified U-Net**:
- Encoder: Conv(32) → Conv(32) → MaxPool  
  Conv(64) → Conv(64) → MaxPool  
  Conv(128) → Conv(128) → MaxPool
- Bottleneck: Conv(256) → Conv(256)
- Decoder: UpSample + Skip Connections + Conv blocks
- Output: **1 channel + Sigmoid activation** (binary segmentation)

**What’s different from “ordinary U-Net”?**
- Lighter (fewer stages than full U-Net variants)
- No pretrained backbone (trained from scratch)
- Output is **binary** (1 channel) instead of multi-class

---

## Training Setup
- Epochs: 75
- Batch size: 2 (GPU memory constraint)
- Loss: Binary Cross Entropy
- Optimizer: Adam (1e-3)
- Metrics: Accuracy, Precision, Recall

---

## Evaluation Metrics
- Pixel Accuracy
- Precision / Recall / F1
- IoU (Lane)
- Dice (Lane)
- Confusion Matrix (pixel-level)

---

## How To Run

### 1) Create Environment & Install
```bash
conda create -n raviole python=3.10 -y
conda activate raviole
pip install -r requirements.txt
```
### 2) Train (U-Net)
```bash
python src/train_unet.py --data_dir lane_detection --img_size 256 --epochs 75 --batch 2
```
### 3) Evaluate (Confusion Matrix + IoU/Dice)
```bash
python src/evaluate_metrics.py --model models/lane_model.h5 --val_dir lane_detection/valid
```
### 4) Stereo Rectification + Steering (Live)
```bash
python src/stereo_rectify_and_steer.py --model models/lane_model.h5 --calib calibration/stereo_calibration.npz
```

Authors

1. Alvito Danendra Putra

2. Ben Arthur

3. Nathan Shaun

4. William Henry
