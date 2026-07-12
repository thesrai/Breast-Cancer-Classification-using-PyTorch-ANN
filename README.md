# Breast-Cancer-Classification-using-PyTorch-ANN
A simple Artificial Neural Network (ANN) implemented in PyTorch for binary classification on the Breast Cancer Wisconsin dataset. This project demonstrates the complete deep learning workflow, including data preprocessing, model implementation, training, and evaluation.

## Dataset
- Breast Cancer Wisconsin Dataset
- Source: scikit-learn
- Binary Classification
- 30 numerical features
- Target:
0 = Malignant
1 = Benign

## Technologies
- Python
- PyTorch
- scikit-learn

## Workflow
- Load Breast Cancer Dataset
- Train/Test Split
- Feature Standardization
- Convert to PyTorch Tensors
- Create Custom Dataset
- Create DataLoader
- Build ANN Model
- Train the Model
- Evaluate the Model
- Calculate Accuracy
- Predict on New Sample

## Model Architecture
- Input Layer (30)
- Linear (30 → 32)
- ReLU
- Linear (32 → 1)
- Sigmoid

## Trainig
- Loss Function: Binary Cross Entropy Loss (BCELoss)
- Optimizer: Adam
- Learning Rate: 0.001
- Epochs: 100

## Example Output
Accuracy: 0.6053
Malignant
tensor([[0.0011]])

پ


