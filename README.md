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

### 2. **Neural Network Implementation (Scratch & Keras)** (`notebooks/Neural Network Implementation from Scratch and with Keras on MNIST.ipynb`)  
   - Implemented a simple **neural network from scratch** using **NumPy** (sigmoid activation, forward & backward propagation, SGD).  
   - Trained the scratch-built model on a **small synthetic dataset** to validate learning.  
   - Preprocessed and loaded the **MNIST dataset** for digit recognition.  
   - Built and trained a **Keras-based neural network** for handwritten digit classification.  
   - Tools: Python, NumPy, TensorFlow/Keras  
   - Example: trained model achieved high accuracy on MNIST with Keras implementation.

---

### 3.**MNIST Digit Classification (Deep Neural Network)** (`notebooks/MNIST Digit Classification Neural Network.ipynb`)  
   - Implemented a **fully connected neural network** for handwritten digit classification using the MNIST dataset.  
   - **Data preprocessing**: pixel normalization + one-hot encoding of labels.  
   - **Model architecture**:  
     - Input layer: 784 neurons (28×28 pixels)  
     - Hidden layers: 3 layers × 128 neurons each (ReLU activation)  
     - Output layer: 10 neurons (softmax)  
   - **Training setup**:  
     - Optimizer: Adam  
     - Loss: Categorical Crossentropy  
     - Epochs: 10  
     - Batch size: 20  
   - **Performance**:  
     - Final accuracy: ~99.4%  
     - Parameters: 328,160 total (109,386 trainable)  
   - Tools: TensorFlow/Keras, NumPy  
   - Successfully classifies handwritten digits with very high accuracy. 
---

### 4. **Image Classification with Transfer Learning (VGG16)** (`notebooks/VGG_for_classification.ipynb`)  
   - Built a **10-class image classification model** using **transfer learning** with VGG16.  
   - Dataset: 10 categories (c0–c9), split into train/validation/test sets.  

**Key Steps:**  
- **Setup & Paths**: Configured dataset paths, set random seeds.  
- **Data Loading**: Loaded metadata (CSV), verified image paths, and visualized class distribution.  
- **Data Splitting**: Train (80%), Validation (10%), Test (10%).  
- **Data Augmentation**: Rotation, shifting, flipping, normalization (via `ImageDataGenerator`).  
- **Model Building**:  
  - Base: VGG16 pre-trained on ImageNet (frozen layers).  
  - Custom Head: Flatten → Dense(256, ReLU) → Dropout(0.5) → Dense(10, Softmax).  
  - Loss: categorical crossentropy | Optimizer: Adam.  
- **Training**:  
  - 30 epochs with **early stopping** and **model checkpointing**.  
  - Achieved ~94% validation accuracy.  
- **Fine-Tuning**:  
  - Unfroze last 15 layers of VGG16.  
  - Trained 10 more epochs with LR = 1e-5.  
  - Achieved ~98% validation accuracy.  
- **Model Saving & Evaluation**: Saved as `final_model.keras`, evaluated successfully on test set.  

**Key Techniques:**  
- Transfer learning with VGG16  
- Data augmentation  
- Learning rate scheduling  
- Fine-tuning pre-trained models  
- Callbacks (early stopping, checkpointing)  

*Tools: TensorFlow/Keras, Pandas, Matplotlib*    
