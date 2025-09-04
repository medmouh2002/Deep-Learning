# 🤖 Deep Learning Module

This folder contains projects related to **deep learning** — from foundational implementations to applied CNN models for real-world tasks.

## Projects

### 1. **Linear Regression from Scratch** (`notebooks/linear_regression.ipynb`)  
   - Built a single-neuron linear regression model to predict Fahrenheit temperatures from Celsius inputs.  
   - Implemented gradient descent manually without high-level frameworks.  
   - Steps:
     - Data generation (160 Celsius–Fahrenheit pairs)  
     - Model initialization (random weight & bias)  
     - Forward pass (predictions + MSE loss)  
     - Backward pass (manual gradient calculation)  
     - Parameter update (gradient descent loop)  
     - Validation with train–test split  
   - Tools: Python, NumPy, Matplotlib  

---

### 2. **Driver Drowsiness Detection** (`notebooks/driver_drowsiness_detection.ipynb`)  
   - Vision-based **CNN model** to detect drowsy drivers from webcam images.  
   - Applied **transfer learning (VGG16)** for feature extraction.  
   - Achieved ~89% accuracy on test data.  
   - Tools: TensorFlow/Keras, OpenCV  
   - Example:  
     ![Drowsiness Demo](images/drowsiness_demo.png)  

---

### 3. **MNIST Digit Recognition** (`notebooks/mnist_digit_recognition.ipynb`)  
   - Implemented a **CNN from scratch** for handwritten digit classification.  
   - Compared performance against a baseline MLP model.  
   - Achieved >98% accuracy on the MNIST dataset.  
   - Tools: TensorFlow, Matplotlib  
   - Example:  
     ![MNIST Results](images/mnist_results.png)  

---

### 4. **CNN vs MLP Comparison** (`notebooks/cnn_vs_mlp_comparison.ipynb`)  
   - Compared performance of **Convolutional Neural Networks** vs traditional fully-connected networks.  
   - Analyzed training curves, overfitting, and generalization.  
   - Tools: TensorFlow, Keras, Matplotlib  
   - Example:  
     ![Training Curves](images/training_curves.png)  
