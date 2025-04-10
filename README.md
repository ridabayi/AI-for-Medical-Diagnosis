# AI-for-Medical-Diagnosis  
*Chest X-Ray Multi-Disease Diagnosis with Deep Learning*

AI-powered detection of 14 thoracic diseases from chest X-ray images using Convolutional Neural Networks (CNN).

<p align="center">
  <img src="https://github.com/ridabayi/AI-for-Medical-Diagnosis/blob/main/xray-header-image.png" alt="Chest X-Ray Sample" width="60%">
</p>

<p align="center">
  <a href="https://arxiv.org/abs/1705.02315"><img src="https://img.shields.io/badge/Dataset-ChestX--ray14-blue"></a>
  <img src="https://img.shields.io/badge/Python-3.8+-blue">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

---

## 📌 Overview

This project develops an advanced deep learning model to classify **14 thoracic diseases** from chest X-ray images. Using a CNN, the model assists medical professionals by providing quick and reliable diagnostic support.

**Goal:** Early detection of critical conditions, supporting faster interventions and better patient outcomes.

---

## 🩻 Targeted Diseases

The model predicts the following conditions:

- Cardiomegaly
- Emphysema
- Effusion
- Hernia
- Infiltration
- Mass
- Nodule
- Atelectasis
- Pneumothorax
- Pleural Thickening
- Pneumonia
- Fibrosis
- Edema
- Consolidation

**Dataset source:** [ChestX-ray14](https://arxiv.org/abs/1705.02315)

---

## 🚀 Project Highlights

- ✅ Multi-class classification: **14 diseases**
- ✅ Data preprocessing & augmentation
- ✅ Custom CNN architecture with TensorFlow/Keras
- ✅ Training visualizations: accuracy & loss curves
- ✅ Potential for real-world deployment in healthcare environments

---

## 🧩 Model Architecture

<p align="center">
  <img src="https://github.com/ridabayi/AI-for-Medical-Diagnosis/blob/main/densenet.png" alt="Densenet" width="60%">
</p>


## 📊 Results Preview

<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/commons/4/4f/Loss_and_accuracy_plot.svg" alt="Training Metrics" width="70%">
</p>

**Training Accuracy:** High convergence over epochs  
**Validation Performance:** Good generalization with minimal overfitting  
**Loss Curves:** Smooth learning progression

---

## 💻 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the notebook
```bash
jupyter notebook C1W1_A1_Chest\ X-Ray\ Medical\ Diagnosis\ with\ Deep\ Learning.ipynb
```

---

## 🧩 Project Structure

```
├── data/               # Chest X-ray images
├── notebook/           # Jupyter notebook with full training pipeline
├── models/             # Saved trained models
├── README.md           # Project documentation
├── requirements.txt    # Python dependencies
└── LICENSE             # MIT License
```

---

## 📈 Improvements & Next Steps

- 🔍 Integrate Grad-CAM visualizations for interpretability
- 🚀 Explore transfer learning with ResNet or DenseNet
- 🌐 Deploy model as a web application (Flask/Streamlit)
- 🧪 Hyperparameter tuning for better accuracy
- ☁️ Test deployment on cloud platforms (AWS, GCP)

---

## 🤝 Acknowledgements

- Dataset: [ChestX-ray14](https://arxiv.org/abs/1705.02315)

---

<p align="center">
  Made with ❤️ for advancing AI in healthcare.
</p>
