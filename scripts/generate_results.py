"""
Generate result visualizations from validation metrics.
Creates confusion matrix, metrics chart, and error analysis plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json
import os

# Validation results
y_true = np.array([0]*298 + [1]*804)  # 298 safe, 804 unsafe
y_pred = np.array([0]*293 + [1]*5 + [0]*11 + [1]*793)

os.makedirs('experiments', exist_ok=True)

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Safe', 'Unsafe'], 
            yticklabels=['Safe', 'Unsafe'],
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - ResNet50 Vision Risk Classifier', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('experiments/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved confusion_matrix.png")
plt.close()

# 2. Metrics Bar Chart
metrics = {
    'Accuracy': 98.55,
    'Precision': 99.37,
    'Recall': 98.63,
    'F1-Score': 99.0,
    'Balanced Acc': 98.48
}

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(metrics.keys(), metrics.values(), 
              color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'])
ax.set_ylim([95, 100])
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.axhline(y=98, color='gray', linestyle='--', alpha=0.5)
ax.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('experiments/metrics_chart.png', dpi=300, bbox_inches='tight')
print("✓ Saved metrics_chart.png")
plt.close()

# 3. Save metrics JSON
results = {
    "model": "ResNet50 (Transfer Learning)",
    "validation_samples": 1102,
    "confusion_matrix": {
        "true_negatives": 293,
        "false_positives": 5,
        "false_negatives": 11,
        "true_positives": 793
    },
    "metrics": {
        "accuracy": 98.55,
        "precision_unsafe": 99.37,
        "recall_unsafe": 98.63,
        "f1_score": 99.0,
        "false_positive_rate": 1.68,
        "balanced_accuracy": 98.48
    },
    "threshold": 0.5
}

with open('experiments/metrics.json', 'w') as f:
    json.dump(results, f, indent=2)
print("✓ Saved metrics.json")
