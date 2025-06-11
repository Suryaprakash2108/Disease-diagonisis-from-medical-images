# 🧬 **Lung and Colon Cancer Classification Using CNN – Deep Learning Project**  

This project uses deep learning to classify histopathological images of lung and colon cancer. Leveraging a powerful CNN architecture with InceptionResNetV2 as the backbone, it aims to automate cancer diagnosis and assist healthcare professionals with high accuracy.

🔬 Built as part of an academic project by Mopada Surya Prakash, UG Scholar at Mohan Babu University, Tirupati, India.

---

## 🧭 Project Overview

**Problem Statement:** Detecting cancer in tissue samples is challenging and time-consuming. Traditional manual diagnosis methods can be inconsistent.  

**Solution:** This project uses CNNs to identify cancerous tissues in medical images. It implements transfer learning with InceptionResNetV2 and a custom classifier to distinguish between lung and colon cancer types.  

**Use Case:** Useful for researchers, pathologists, and AI-based diagnostic systems to improve cancer detection accuracy and speed.

---

## 📂 Files Included

- `lung-and-colon-cancer.ipynb`: Main notebook for training and evaluating the CNN model  
- `dataset/`: Directory containing labeled images of lung and colon tissues  
- `outputs/`: Contains plots, trained model, and performance metrics  
- `README.md`: This documentation  

---

## 🧱 Model Architecture

- InceptionResNetV2 (pretrained on ImageNet) for feature extraction  
- Dense(512) → Dropout(0.2) → Dense(512) → Dropout(0.2)  
- Final Dense layer with softmax for classification  
- **Optimizer:** Adam  
- **Loss Function:** Sparse Categorical Crossentropy  

---

## 📈 Model Performance

| Metric             | Value     |
|--------------------|-----------|
| **Train Accuracy** | 92.78%    |
| **Validation Accuracy** | 91.52%    |
| **Test Accuracy**  | 91.84%    |
| **Test Loss**      | 182.54    |

> The model generalizes well across datasets and achieves strong classification accuracy.

---

## 🖼️ Output

The model predicts the class of a histopathological image (lung or colon cancer) based on visual patterns.  
Performance plots and prediction functions are included in the notebook.

---

## 🛠️ Tech Stack

- Python  
- TensorFlow / Keras  
- NumPy, Pandas, OpenCV, PIL  
- Matplotlib, Seaborn  
- Jupyter Notebook  

---
