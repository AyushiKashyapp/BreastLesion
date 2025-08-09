# 🎗️ Breast Lesion Classification on CBIS-DDSM Dataset

Welcome to this breast cancer classification project! Here, we use mammogram images to classify breast lesions as **Benign** or **Malignant** using deep learning. [Project Markdown](https://github.com/AyushiKashyapp/BreastLesion/blob/main/BreastLesion.ipynb)

---

## 🗂️ Dataset

We use the **CBIS-DDSM** dataset — a collection of mammogram images with expert annotations and pathology labels. The dataset is downloaded directly using the Kaggle API (via `kagglehub`), and preprocessed to extract cropped images with accurate file paths.

[Find it here!](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)

---

## 🚀 Project Workflow

### 1️⃣ Data Loading & Cleaning

- Load train and test metadata CSVs.
- Fix image paths so we can load the correct mammogram images.
- Standardize column names for consistency.
- Create a binary label:
  - **Benign** (label 0)
  - **Malignant** (label 1)
- Handle missing values in categorical columns by filling them with the most common value.
- One-hot encode these categorical features (optional for tabular analysis).
- Build a custom PyTorch Dataset class to load images and labels easily.

### 2️⃣ Image Preprocessing & Augmentation

- Resize all images to 224x224 pixels (standard input size for CNNs).
- Apply random horizontal flips and normalize images with ImageNet stats during training.
- For validation/testing, just resize and normalize without augmentation.

### 3️⃣ Model Setup

- Use a pretrained **ResNet18** model to leverage learned features.
- Replace the final layer for binary classification.
- Use binary cross-entropy loss with logits for stable training.
- Use Adam optimizer with a learning rate of 0.0001.

### 4️⃣ Training & Evaluation

- Train for 5 epochs (short but enough to demonstrate the pipeline).
- Track training loss and test accuracy after each epoch.
- Evaluate performance using:
  - Accuracy
  - Precision, Recall, F1-score (classification report)
  - Confusion matrix visualization

### 5️⃣ Explainability with Grad-CAM 🔥

- Implement Grad-CAM to understand which parts of the mammogram influenced the model’s decision.
- Overlay heatmaps on images to visually highlight important regions.
- Helps in building trust and understanding of model predictions.

---

## 📊 Results Summary

| Metric       | Value (after 5 epochs)     |
|--------------|----------------------------|
| Accuracy     | ~50%                       |
| Precision    | 70% (Benign), 42% (Malignant)  |
| Recall       | 33% (Benign), 78% (Malignant)  |
| F1-Score     | 45% (Benign), 55% (Malignant)  |

> **Note:** Training for only 5 epochs limits performance. This project focuses on building a reproducible pipeline with visual insights rather than state-of-the-art accuracy.

---

## 💡 What’s Next? Ideas to Improve

- Train longer (more epochs) and tune learning rate.
- Try more powerful models or ensembles.
- Add more augmentations for robust learning.
- Combine image data with clinical features for multimodal analysis.
- Enhance Grad-CAM visualizations for clearer explanations.

---

## ⚙️ Dependencies

- Python 3.x
- PyTorch & torchvision
- OpenCV
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- Pillow (PIL)

---
