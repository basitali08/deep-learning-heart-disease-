# 🧠 Deep Learning for Heart Disease Prediction — PyTorch

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-F7931E?logo=scikit-learn)](https://scikit-learn.org)

Neural network implementation for heart disease prediction using **PyTorch** with BatchNorm, Dropout, and learning rate scheduling. Compared against Random Forest and Logistic Regression baselines.

---

## Architecture

| Layer | Type | Size | Regularization |
|-------|------|:----:|:--------------:|
| Input | Dense | 13 → 64 | — |
| Hidden 1 | Dense + BatchNorm + ReLU + Dropout | 64 → 32 | Dropout 0.3 |
| Hidden 2 | Dense + BatchNorm + ReLU + Dropout | 32 → 16 | Dropout 0.2 |
| Output | Dense + Sigmoid | 16 → 1 | — |

- **Optimizer**: Adam (lr=0.001)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: BCELoss
- **Epochs**: 100 (with early stopping)
- **Weight Decay**: 1e-4

## Results

| Model | Accuracy |
|-------|:--------:|
| **Neural Network (PyTorch)** | **50.8%** |
| Random Forest | 42.6% |
| Logistic Regression | 36.1% |

## Key Features

- PyTorch `Dataset` + `DataLoader` for batch training
- Training/validation loss & accuracy curves
- Learning rate scheduling for convergence
- Permutation-based feature importance
- Model comparison with sklearn baselines

## Project Structure

```
deep-learning-heart-disease/
├── src/
│   └── heart_disease_dl.py       # Complete PyTorch pipeline
├── data/                          # UCI Heart Disease dataset
├── models/                        # Saved model weights
├── results/                       # Training curves + metrics
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install torch pandas numpy matplotlib seaborn scikit-learn
python src/heart_disease_dl.py
```

---

<p align="center">
<b>Built by Basit Ali</b> · <a href="https://github.com/basitali08">GitHub</a> · <a href="mailto:whoisbasit@gmail.com">Email</a>
</p>
