# 🤖 Deep Learning 

This repository demonstrates both **fundamental concepts** (manual gradient descent, linear regression) and **practical applications** (CNNs, transfer learning) for computer vision and prediction tasks.  

---

## 📂 Contents

### 1. Linear Regression from Scratch (`notebooks/linear_regression.ipynb`)  
- A **hands-on implementation** of a single-neuron linear regression model to predict Fahrenheit temperatures from Celsius inputs.  
- Demonstrates a fundamental understanding of **deep learning mechanics** by building gradient descent **without high-level frameworks**.  

**Key Features:**  
- **Data Generation**: Created a synthetic dataset of 160 Celsius–Fahrenheit pairs.  
- **Model Initialization**: Randomly initialized parameters (weight & bias).  
- **Core Algorithm Implementation**:  
  - Forward Pass → compute predictions & Mean Squared Error (MSE) loss  
  - Backward Pass → manually calculate gradients for weight & bias  
  - Parameter Update → gradient descent loop for optimization  
- **Validation**: Evaluated performance using a train–test split.  

*Tools: Python, NumPy, Matplotlib*  

---

### 2. Driver Drowsiness Detection (`notebooks/driver_drowsiness_detection.ipynb`)  
- Built a **CNN-based vision system** to detect drowsy drivers from webcam images.  
- Applied **transfer learning** (VGG16) for feature extraction.  
- Achieved ~89% classification accuracy on test data.  
- Tools: TensorFlow/Keras, OpenCV  
- Example:  
  ![Drowsiness Demo](images/drowsiness_demo.png)  

---

### 3. MNIST Digit Recognition (`notebooks/mnist_digit_recognition.ipynb`)  
- Implemented a CNN from scratch for handwritten digit classification.  
- Compared performance with a baseline MLP model.  
- Achieved >98% accuracy on MNIST dataset.  
- Tools: TensorFlow, Matplotlib  
- Example:  
  ![MNIST Results](images/mnist_results.png)  

---

### 4. CNN vs MLP Comparison (`notebooks/cnn_vs_mlp_comparison.ipynb`)  
- Compared performance of **Convolutional Neural Networks (CNNs)** vs traditional fully-connected networks.  
- Analyzed training curves, overfitting, and generalization.  
- Visualized learning curves:  
  ![Training Curves](images/training_curves.png)  

---

## 🛠️ Tools & Libraries
- Python  
- TensorFlow, Keras, PyTorch  
- NumPy, Pandas, Matplotlib  
- OpenCV (for image preprocessing)  

---

## 🚀 Next Steps
- Explore **data augmentation** for robust CNN training  
- Experiment with **GANs** for synthetic data generation  
- Deploy trained models as **APIs or Streamlit apps**  

---

## 🔒 Notes
- Datasets (e.g., MNIST, driver monitoring dataset) are **not included** due to size/licensing.  
- This repo contains code, notebooks, and sample outputs only.
