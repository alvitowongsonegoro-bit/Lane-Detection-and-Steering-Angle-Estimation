# Lane-Detection-and-Steering-Angle-Estimation

This project implements a **vision-based lane detection and steering angle estimation system** using **deep learning (U-Net)** and **Kalman Filter–based sensor fusion**.  
It is designed as an academic autonomous vehicle perception and control pipeline.

---

## 📌 Project Overview

Lane detection is a core component of autonomous driving.  
Traditional computer vision methods (edge detection, thresholding) are highly sensitive to:

- Lighting variations  
- Shadows and reflections  
- Complex road textures  

This project uses **deep learning–based semantic segmentation** to robustly extract lane regions and estimate **lane curvature and steering angle**.

---

## 🎯 Objectives

- Perform **lane segmentation** using a custom U-Net CNN
- Extract lane boundaries from segmentation masks
- Estimate **steering angle** from lane curvature
- Apply **Kalman Filter** for noise reduction and smoothing
- Demonstrate a full **perception → estimation → control** pipeline

---

## 🧠 System Pipeline

```text
Stereo Camera
     ↓
Image Preprocessing
     ↓
U-Net Lane Segmentation
     ↓
Lane Mask
     ↓
Boundary Extraction
     ↓
Curve Fitting (2nd order polynomial)
     ↓
Steering Angle Estimation
     ↓
Kalman Filter
     ↓
Smoothed Steering Angle Output
```

📊 Dataset

- Source: Roboflow

  - Data Type: Image–Mask pairs

  - Image: RGB road images

- Mask:

  - White (1): Lane / drivable area

  - Black (0): Background

- Preprocessing

  - Resize to 256 × 256

  - Normalize pixel values to [0, 1]

  - Masks converted to binary format

🏗️ Model Architecture

1. Base Model: Modified U-Net

2. Encoder–Decoder CNN

3. Skip connections preserve lane boundaries

4. Lightweight architecture for faster inference

5. Trained from scratch (no pretrained weights)

- Key Details

  - Output channels: 1 (binary segmentation)

  - Activation: Sigmoid

  - Loss: Binary Cross-Entropy

  - Optimizer: Adam

🏋️ Training Configuration

- Epochs: 75

- Batch size: 2 (GPU memory limitation)

- Metrics:

  1. Accuracy

  2. Precision

  3. Recall

  4. Training Results

      - Accuracy > 99%

      - Stable validation loss

      - No significant overfitting

📐 Steering Angle Estimation

Steps:

1. Extract lane boundaries from mask

2. Fit 2nd-order polynomial

3. Compute tangent angle relative to vehicle

4. Apply Kalman Filter for smoothing

5. Kalman Filter

  - Reduces noise between frames

  - Produces stable steering commands

Output can be mapped to:

- Steering angle (degrees)

- Steering voltage (MCU / EPS)

⚙️ Hardware Integration (Concept)

- Stereo Camera (Left–Right)

- Microcontroller: ESP32

- Actuator: Stepper Motor (NEMA 23) + TB6600 driver

- Control output: PWM steering signal

⚠️ Limitations

- No explicit data augmentation

- Binary segmentation only (no left/right lane separation)

- Performance drops on wet roads due to reflections

- EPS control not fully synchronized

🚀 Future Work

- Add data augmentation (brightness, blur, shadows)

- Multi-class lane segmentation

- Real-time optimization

- Full closed-loop EPS control

- Better stereo depth utilization

👨‍💻 Authors

1. Alvito Danendra Putra

2. Ben Arthur

3. Nathan Shaun

4. William Henry
