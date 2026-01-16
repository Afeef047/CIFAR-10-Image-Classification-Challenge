# CIFAR-10 Image Classification Challenge

This repository contains a multi-level submission for the CIFAR-10 image classification challenge. The objective was to build a robust classifier, starting from a baseline transfer learning model and progressively improving accuracy through advanced data augmentation, architectural changes, and ensemble methods.

## Dataset Overview

The **CIFAR-10** dataset consists of 60,000  color images in 10 classes, with 6,000 images per class.

* **Split Strategy:** 80% Train | 10% Validation | 10% Test
* **Classes:** *airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.*

---

## Level-wise Performance Summary

| Level | Strategy | Backbone | Key Techniques | Test Accuracy |
| --- | --- | --- | --- | --- |
| **Level 1** | **Baseline** | ResNet-18 | Transfer Learning, Light Augmentation | **0.8896** |
| **Level 2** | **Intermediate** | ConvNeXt-Tiny | Strong Augmentation, LR Scheduler | **0.9234** |
| **Level 3** | **Custom** | ConvNeXt-Tiny | Custom MLP Head, Partial Fine-Tuning | **0.9110** |
| **Level 4** | **Ensemble** | **L2 + L3** | **Weighted Soft-Probability Ensemble** | **0.9309** |

---

## Tools & Environment

* **Framework:** PyTorch
* **Libraries:** `torchvision`, `timm` (for ConvNeXt), `matplotlib`, `seaborn`
* **Optimization:** Automatic Mixed Precision (AMP) for faster training
* **Hardware:** Google Colab (NVIDIA T4 GPU)

---

## Repository Structure

```text
├── models/               # Saved model weights (.pth)
│   ├── level1.pth
│   ├── level2.pth
│   └── level3.pth
├── results/              # Evaluation artifacts
│   ├── curves/           # Training loss and accuracy plots
│   ├── confusion_matrix.png
│   └── ensemble_plot.png
├── logs/                 # Execution history
│   └── training.log      # Detailed logs of training/validation per epoch
├── CIFAR_10.ipynb        # Main implementation notebook
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation

```

---

## Observations & Insights

* **Augmentation Impact:** Moving from basic flips to stronger augmentations in Level 2 significantly reduced the generalization gap, pushing accuracy past 92%.
* **Architectural Trade-offs:** The custom MLP head in Level 3 provided better class-specific granularity, though it required more careful regularization to compete with the frozen backbone approach of Level 2.
* **Ensemble Power:** By combining the high-performing Level 2 model with the architectural variety of Level 3, the Level 4 ensemble achieved a peak accuracy of **93.09%**.

---

## Running Instructions

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/CIFAR10-Challenge.git
cd CIFAR10-Challenge

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Execute the Pipeline**:
Open `CIFAR_10.ipynb` in Google Colab or Jupyter Notebook and run all cells.
4. **Monitor Performance**:
Refer to `logs/training.log` for a step-by-step breakdown of training metrics.

