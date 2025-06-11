Lung and Colon Cancer Classification

This deep learning project focuses on the classification of histopathological images of lung and colon tissues using Convolutional Neural Networks (CNNs). It supports medical diagnostics by identifying cancerous and non-cancerous tissues with high accuracy.

Project Structure

lung-and-colon-cancer/
│
├── lung-and-colon-cancer.ipynb - Main notebook for training and evaluation
├── README.txt - This documentation
├── dataset/
│ ├── train/ - Training images (organized by class)
│ ├── val/ - Validation images
│ └── test/ - Test images
├── outputs/
│ ├── model/ - Saved trained model
│ └── plots/ - Accuracy/loss graphs and visualizations

Model Results

Train Accuracy: 92.78%
Validation Accuracy: 91.52%
Test Accuracy: 91.84%
Test Loss: 182.54

The model shows high performance and generalization capability, making it suitable for experimental diagnostic support.

Dataset Details

Includes histopathological images of lung and colon tissues

Images are categorized and stored in class-specific folders

Separate folders are maintained for training, validation, and testing

Technologies Used

Python

TensorFlow and Keras

NumPy and Pandas

OpenCV and PIL

Matplotlib and Seaborn

Jupyter Notebook / Google Colab

How to Run

Open the notebook in Jupyter or Google Colab

Upload the dataset into the dataset/train, dataset/val, and dataset/test folders

Execute all notebook cells

View training metrics, evaluation results, and prediction examples

Key Features

CNN model with convolution, pooling, dropout, and dense layers

Accurate multi-class classification

Visualization of model training progress

Code optimized for educational and research use

Disclaimer

This model is intended for educational and research purposes only. It is not approved for clinical or diagnostic use in medical settings.
