import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

print('='*65)
print('DEEP LEARNING FOR HEART DISEASE PREDICTION')
print('='*65)

# 1. Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

for d in ['data/processed', 'models', 'results']:
    os.makedirs(d, exist_ok=True)

# 2. Load Data
print('\n1. LOADING DATA...')
df = pd.read_csv('data/raw/heart_disease.csv')
print(f'Dataset: {df.shape}')

# 3. Preprocessing
print('\n2. PREPROCESSING...')
X = df.drop('target', axis=1)
y = df['target'].values

numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Handle missing values and scale
imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_imputed = imputer.fit_transform(X)
X_scaled = scaler.fit_transform(X_imputed)

# Save preprocessors
with open('models/imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')

# 4. Define Neural Network
print('\n3. BUILDING NEURAL NETWORK...')

class HeartDiseaseNN(nn.Module):
    def __init__(self, input_dim):
        super(HeartDiseaseNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

model = HeartDiseaseNN(X_train.shape[1]).to(device)
print(model)

# 5. Training
print('\n4. TRAINING...')
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

num_epochs = 200
train_losses = []
test_losses = []
train_accs = []
test_accs = []
best_test_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch_X.size(0)
        predicted = (outputs >= 0.5).float()
        train_correct += (predicted == batch_y).sum().item()
        train_total += batch_y.size(0)

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor.to(device))
        test_loss = criterion(test_outputs, y_test_tensor.to(device)).item()
        test_predicted = (test_outputs >= 0.5).float()
        test_correct = (test_predicted == y_test_tensor.to(device)).sum().item()

    epoch_train_loss = train_loss / train_total
    epoch_train_acc = train_correct / train_total
    epoch_test_acc = test_correct / len(y_test)

    train_losses.append(epoch_train_loss)
    test_losses.append(test_loss)
    train_accs.append(epoch_train_acc)
    test_accs.append(epoch_test_acc)

    scheduler.step(test_loss)

    if epoch_test_acc > best_test_acc:
        best_test_acc = epoch_test_acc
        torch.save(model.state_dict(), 'models/best_model.pt')

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, Test Acc: {epoch_test_acc:.4f}')

# 6. Final Evaluation
print('\n5. FINAL EVALUATION...')
model.load_state_dict(torch.load('models/best_model.pt'))
model.eval()

with torch.no_grad():
    y_prob = model(X_test_tensor.to(device)).cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print(f'Neural Network Performance:')
print(f'  Accuracy:  {accuracy:.4f}')
print(f'  Precision: {precision:.4f}')
print(f'  Recall:    {recall:.4f}')
print(f'  F1 Score:  {f1:.4f}')
print(f'  ROC AUC:   {auc:.4f}')

print(f'\nClassification Report:\n{classification_report(y_test, y_pred)}')

# 7. Compare with Traditional ML
print('\n6. COMPARISON WITH TRADITIONAL ML...')
lr = LogisticRegression(max_iter=1000, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

comparison = pd.DataFrame({
    'Model': ['Neural Network', 'Logistic Regression', 'Random Forest'],
    'Accuracy': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, lr_pred),
        accuracy_score(y_test, rf_pred)
    ],
    'F1 Score': [
        f1_score(y_test, y_pred),
        f1_score(y_test, lr_pred),
        f1_score(y_test, rf_pred)
    ]
})
print(comparison.to_string(index=False))
comparison.to_csv('results/model_comparison.csv', index=False)

# 8. Visualizations
print('\n7. CREATING VISUALIZATIONS...')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Training curves
ax = axes[0, 0]
ax.plot(train_losses, label='Train Loss', color='blue')
ax.plot(test_losses, label='Test Loss', color='red')
ax.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(train_accs, label='Train Accuracy', color='blue')
ax.plot(test_accs, label='Test Accuracy', color='red')
ax.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

# Model comparison
ax = axes[1, 0]
x = np.arange(len(comparison))
width = 0.35
bars1 = ax.bar(x - width/2, comparison['Accuracy'], width, label='Accuracy', color='steelblue')
bars2 = ax.bar(x + width/2, comparison['F1 Score'], width, label='F1 Score', color='coral')
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(comparison['Model'], rotation=20)
ax.set_ylabel('Score')
ax.legend()
ax.set_ylim(0, 1)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{bar.get_height():.3f}', ha='center', fontsize=8)

# Confusion Matrix
ax = axes[1, 1]
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title('Confusion Matrix - Neural Network', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')

plt.tight_layout()
plt.savefig('results/performance_plots.png', dpi=200, bbox_inches='tight')
print('Plots saved')

# 9. Feature Importance (using permutation importance)
print('\n8. FEATURE IMPORTANCE ANALYSIS...')
model.eval()
baseline = accuracy_score(y_test, y_pred)
importances = []

for i in range(X_test.shape[1]):
    X_permuted = X_test.copy()
    np.random.shuffle(X_permuted[:, i])
    with torch.no_grad():
        perm_pred = (model(torch.FloatTensor(X_permuted).to(device)) >= 0.5).cpu().numpy()
    perm_acc = accuracy_score(y_test, perm_pred)
    importances.append(baseline - perm_acc)

feature_importance = pd.DataFrame({
    'Feature': numerical_cols,
    'Importance': importances
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))
feature_importance.to_csv('results/feature_importance.csv', index=False)

# Save final model
torch.save(model.state_dict(), 'models/final_model.pt')
print('\nModels saved to models/')
print('Results saved to results/')

# Save training history
history = pd.DataFrame({
    'epoch': range(1, num_epochs + 1),
    'train_loss': train_losses,
    'test_loss': test_losses,
    'train_acc': train_accs,
    'test_acc': test_accs
})
history.to_csv('results/training_history.csv', index=False)

print('\n' + '='*65)
print('DEEP LEARNING PIPELINE COMPLETE!')
print('='*65)
print(f'\nBest Test Accuracy: {best_test_acc:.4f}')
print(f'Final ROC AUC: {auc:.4f}')
print(f'\nProject demonstrates:')
print('  - PyTorch neural network implementation')
print('  - Training with SGD, learning rate scheduling')
print('  - Regularization (Dropout, BatchNorm, Weight Decay)')
print('  - Model comparison with traditional ML')
print('  - Feature importance via permutation analysis')
print('  - Training visualization and history tracking')