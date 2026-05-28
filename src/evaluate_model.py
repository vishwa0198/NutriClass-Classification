import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------------------
# Paths
# -------------------------------
models_dir = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models"
data_path = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_final.csv"
plots_dir = os.path.join(models_dir, "plots")
evaluation_dir = os.path.join(models_dir, "evaluation")
os.makedirs(evaluation_dir, exist_ok=True)

print("🔍 NUTRICLASS MODEL EVALUATION")
print("=" * 50)

# -------------------------------
# Load model and data
# -------------------------------
try:
    best_model = joblib.load(os.path.join(models_dir, "model.pkl"))
    print("✅ Best model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found. Please run train_models.py first.")
    exit()

try:
    le = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))
    print("✅ Label encoder loaded successfully")
except FileNotFoundError:
    print("⚠️ Label encoder not found. Using default class names.")
    le = None

# Load data
df = pd.read_csv(data_path)
feature_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
                'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
X = df[feature_cols]
y = df['Food_Name_encoded']

print(f"📊 Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
print(f"🎯 Number of classes: {len(np.unique(y))}")

# Get class names
if le is not None:
    class_names = le.classes_
else:
    class_names = [f"Class_{i}" for i in sorted(np.unique(y))]

print(f"📋 Classes: {', '.join(class_names)}")

# -------------------------------
# Split data for proper evaluation
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Make predictions
# -------------------------------
print("\n🔮 Making predictions...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)

# Also evaluate on training set to check overfitting
y_train_pred = best_model.predict(X_train)

# -------------------------------
# Basic Metrics
# -------------------------------
print("\n📈 BASIC PERFORMANCE METRICS")
print("-" * 40)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_pred)
overfitting_score = train_accuracy - test_accuracy

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Overfitting Score: {overfitting_score:.4f}")

if overfitting_score > 0.05:
    print("⚠️ Model shows signs of overfitting")
elif overfitting_score < -0.05:
    print("⚠️ Model might be underfitting")
else:
    print("✅ Model shows good generalization")

# -------------------------------
# Detailed Classification Report
# -------------------------------
print("\n📊 DETAILED CLASSIFICATION REPORT")
print("-" * 50)
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
print(classification_report(y_test, y_pred, target_names=class_names))

# Convert report to DataFrame for better analysis
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(evaluation_dir, "classification_report.csv"))

# -------------------------------
# Per-class Performance Analysis
# -------------------------------
print("\n🎯 PER-CLASS PERFORMANCE ANALYSIS")
print("-" * 50)

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)

performance_df = pd.DataFrame({
    'Class': class_names,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Support': support
})

print(performance_df.round(4))
performance_df.to_csv(os.path.join(evaluation_dir, "per_class_performance.csv"), index=False)

# Find best and worst performing classes
best_f1_idx = np.argmax(f1)
worst_f1_idx = np.argmin(f1)

print(f"\n🏆 Best performing class: {class_names[best_f1_idx]} (F1: {f1[best_f1_idx]:.4f})")
print(f"⚠️ Worst performing class: {class_names[worst_f1_idx]} (F1: {f1[worst_f1_idx]:.4f})")

# -------------------------------
# Confusion Matrix Visualization
# -------------------------------
print("\n📋 Creating confusion matrix...")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = confusion_matrix(y_test, y_pred, normalize='true')

# Create subplots for both raw and normalized confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0])
axes[0].set_title('Confusion Matrix (Raw Counts)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
confusion_matrix_path = os.path.join(evaluation_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Performance Metrics Visualization
# -------------------------------
print("📊 Creating performance metrics visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Per-class metrics bar plot
metrics_data = performance_df.set_index('Class')[['Precision', 'Recall', 'F1-Score']]
metrics_data.plot(kind='bar', ax=axes[0,0], width=0.8)
axes[0,0].set_title('Per-Class Performance Metrics')
axes[0,0].set_ylabel('Score')
axes[0,0].legend()
axes[0,0].tick_params(axis='x', rotation=45)

# 2. Support (sample count) per class
axes[0,1].bar(class_names, support, color='lightcoral', alpha=0.7)
axes[0,1].set_title('Test Samples per Class')
axes[0,1].set_ylabel('Number of Samples')
axes[0,1].tick_params(axis='x', rotation=45)

# 3. Precision vs Recall scatter plot
axes[1,0].scatter(recall, precision, s=support*2, alpha=0.6, c=range(len(class_names)), cmap='viridis')
for i, class_name in enumerate(class_names):
    axes[1,0].annotate(class_name, (recall[i], precision[i]), xytext=(5, 5), 
                      textcoords='offset points', fontsize=8)
axes[1,0].set_xlabel('Recall')
axes[1,0].set_ylabel('Precision')
axes[1,0].set_title('Precision vs Recall (bubble size = support)')
axes[1,0].grid(True, alpha=0.3)

# 4. Model comparison (if available)
try:
    detailed_results = joblib.load(os.path.join(models_dir, "model_comparison_results.pkl"))
    model_names = list(detailed_results.keys())
    test_accs = [detailed_results[name]['test_acc'] for name in model_names]
    
    axes[1,1].barh(model_names, test_accs, color='skyblue', alpha=0.7)
    axes[1,1].set_xlabel('Test Accuracy')
    axes[1,1].set_title('Model Comparison')
    axes[1,1].grid(True, alpha=0.3)
    
    # Highlight best model
    best_idx = np.argmax(test_accs)
    axes[1,1].barh(model_names[best_idx], test_accs[best_idx], color='gold', alpha=0.8)
    
except FileNotFoundError:
    axes[1,1].text(0.5, 0.5, 'Model comparison\ndata not available', 
                   ha='center', va='center', transform=axes[1,1].transAxes)
    axes[1,1].set_title('Model Comparison (Not Available)')

plt.tight_layout()
performance_plot_path = os.path.join(evaluation_dir, "performance_metrics.png")
plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# ROC Curves for Multi-class
# -------------------------------
print("📈 Creating ROC curves...")

# Binarize the output for multi-class ROC
y_test_bin = label_binarize(y_test, classes=np.unique(y))
n_classes = y_test_bin.shape[1]

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(12, 8))

colors = plt.cm.Set3(np.linspace(0, 1, n_classes))

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
             label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curves')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()

roc_curves_path = os.path.join(evaluation_dir, "roc_curves.png")
plt.savefig(roc_curves_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Feature Importance Analysis
# -------------------------------
print("🔍 Analyzing feature importance...")

if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    importance = best_model.named_steps['classifier'].feature_importances_
    
    # Create feature importance DataFrame
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 FEATURE IMPORTANCE RANKING")
    print("-" * 40)
    for idx, row in feature_importance_df.iterrows():
        print(f"{row['Feature']:<20}: {row['Importance']:.4f}")
    
    # Save feature importance
    feature_importance_df.to_csv(os.path.join(evaluation_dir, "feature_importance.csv"), index=False)
    
    # Visualize feature importance
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_cols)))
    bars = plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance Analysis')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance_df['Importance']):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    feature_importance_path = os.path.join(evaluation_dir, "feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Feature importance analysis saved")
    
elif hasattr(best_model.named_steps['classifier'], 'coef_'):
    # For linear models, use coefficients
    coef = best_model.named_steps['classifier'].coef_
    if len(coef.shape) > 1:
        # Multi-class: take mean of absolute coefficients
        importance = np.mean(np.abs(coef), axis=0)
    else:
        importance = np.abs(coef)
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\n🏆 FEATURE IMPORTANCE (from coefficients)")
    print("-" * 40)
    print(feature_importance_df.to_string(index=False))
    
else:
    print("\n⚠️ Feature importance not available for this model type")

# -------------------------------
# Prediction Confidence Analysis
# -------------------------------
print("\n🎯 PREDICTION CONFIDENCE ANALYSIS")
print("-" * 40)

# Get prediction probabilities
max_probs = np.max(y_pred_proba, axis=1)
predicted_classes = np.argmax(y_pred_proba, axis=1)

# Confidence statistics
print(f"Average prediction confidence: {np.mean(max_probs):.4f}")
print(f"Minimum prediction confidence: {np.min(max_probs):.4f}")
print(f"Maximum prediction confidence: {np.max(max_probs):.4f}")

# Low confidence predictions
low_confidence_threshold = 0.5
low_confidence_mask = max_probs < low_confidence_threshold
n_low_confidence = np.sum(low_confidence_mask)

print(f"Predictions with confidence < {low_confidence_threshold}: {n_low_confidence} ({n_low_confidence/len(max_probs)*100:.1f}%)")

# Visualize confidence distribution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(max_probs, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(low_confidence_threshold, color='red', linestyle='--', 
           label=f'Low confidence threshold ({low_confidence_threshold})')
plt.xlabel('Prediction Confidence')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Confidence')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Confidence by class
confidence_by_class = []
for i in range(len(class_names)):
    class_mask = predicted_classes == i
    if np.any(class_mask):
        confidence_by_class.append(max_probs[class_mask])
    else:
        confidence_by_class.append([])

plt.boxplot(confidence_by_class, labels=class_names)
plt.ylabel('Prediction Confidence')
plt.title('Confidence Distribution by Predicted Class')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

plt.tight_layout()
confidence_analysis_path = os.path.join(evaluation_dir, "confidence_analysis.png")
plt.savefig(confidence_analysis_path, dpi=300, bbox_inches='tight')
plt.close()

# -------------------------------
# Error Analysis
# -------------------------------
print("\n❌ ERROR ANALYSIS")
print("-" * 30)

# Find misclassified samples
misclassified_mask = y_test != y_pred
n_misclassified = np.sum(misclassified_mask)

print(f"Total misclassified samples: {n_misclassified} out of {len(y_test)} ({n_misclassified/len(y_test)*100:.1f}%)")

if n_misclassified > 0:
    # Analyze misclassification patterns
    misclass_actual = y_test[misclassified_mask]
    misclass_predicted = y_pred[misclassified_mask]
    misclass_confidence = max_probs[misclassified_mask]
    
    print(f"Average confidence of misclassified samples: {np.mean(misclass_confidence):.4f}")
    
    # Most common misclassification pairs
    misclass_pairs = list(zip(misclass_actual, misclass_predicted))
    from collections import Counter
    common_errors = Counter(misclass_pairs).most_common(5)
    
    print("\nMost common misclassification patterns:")
    for (actual, predicted), count in common_errors:
        print(f"  {class_names[actual]} → {class_names[predicted]}: {count} times")

# -------------------------------
# Save comprehensive evaluation report
# -------------------------------
evaluation_summary = {
    'model_performance': {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'overfitting_score': overfitting_score
    },
    'per_class_metrics': performance_df.to_dict('records'),
    'confusion_matrix': cm.tolist(),
    'roc_auc_scores': {class_names[i]: roc_auc[i] for i in range(len(class_names))},
    'confidence_stats': {
        'mean_confidence': float(np.mean(max_probs)),
        'min_confidence': float(np.min(max_probs)),
        'max_confidence': float(np.max(max_probs)),
        'low_confidence_count': int(n_low_confidence)
    },
    'error_analysis': {
        'total_misclassified': int(n_misclassified),
        'misclassification_rate': float(n_misclassified/len(y_test))
    }
}

# Save as JSON for easy reading
import json
with open(os.path.join(evaluation_dir, "evaluation_summary.json"), 'w') as f:
    json.dump(evaluation_summary, f, indent=2)

print(f"\n✅ EVALUATION COMPLETED!")
print(f"📁 All evaluation results saved in: {evaluation_dir}")
print(f"📊 Generated files:")
print(f"   - confusion_matrix.png")
print(f"   - performance_metrics.png") 
print(f"   - roc_curves.png")
print(f"   - confidence_analysis.png")
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    print(f"   - feature_importance.png")
print(f"   - classification_report.csv")
print(f"   - per_class_performance.csv")
print(f"   - evaluation_summary.json")

print(f"\n🎯 Final Model Performance Summary:")
print(f"   Test Accuracy: {test_accuracy:.4f}")
print(f"   Best Class: {class_names[best_f1_idx]} (F1: {f1[best_f1_idx]:.4f})")
print(f"   Worst Class: {class_names[worst_f1_idx]} (F1: {f1[worst_f1_idx]:.4f})")
print(f"   Average Confidence: {np.mean(max_probs):.4f}")
