# Deep Learning for Heart Disease Prediction

## Overview
Advanced neural network implementation for predicting heart disease using PyTorch. Demonstrates modern deep learning techniques including batch normalization, dropout regularization, learning rate scheduling, and model comparison with traditional machine learning.

## Architecture
- **Input Layer**: 13 clinical features
- **Hidden Layers**: 64 → 32 → 16 neurons
- **Regularization**: BatchNorm, Dropout (0.3, 0.2), Weight Decay
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam with ReduceLROnPlateau scheduler

## Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Neural Network | 50.8% | 51.6% |
| Random Forest | 42.6% | 38.6% |
| Logistic Regression | 36.1% | 40.0% |

## Key Features
- PyTorch `Dataset` and `DataLoader` implementation
- Training/validation loss and accuracy curves
- Learning rate scheduling for optimal convergence
- Permutation-based feature importance analysis
- Model comparison with scikit-learn baselines

## Project Structure
```
deep-learning-heart-disease/
├── data/raw/              # Dataset
├── src/                   # Source code
│   └── heart_disease_dl.py       # Complete DL pipeline
├── models/                # Saved model weights
├── results/               # Plots and evaluation metrics
└── README.md
```

## How to Run
```bash
pip install torch torchvision pandas numpy matplotlib seaborn scikit-learn
python src/heart_disease_dl.py
```

## Technologies
- **PyTorch** - Deep learning framework
- **scikit-learn** - Traditional ML baselines
- **Matplotlib/Seaborn** - Visualization
- **Pandas/NumPy** - Data processing
