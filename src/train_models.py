import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb

# -------------------------------
# Paths
# -------------------------------
data_path = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_final.csv"
models_dir = r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models"
plots_dir = os.path.join(models_dir, "plots")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(data_path)

feature_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 
                'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
X = df[feature_cols]
y = df['Meal_Type_encoded']

print(f"Dataset shape: {X.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}")

# -------------------------------
# Train/Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# Define models with optimized parameters
# -------------------------------
models = {
    "Logistic Regression": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0,
            solver='liblinear',
            penalty='l2',
            multi_class='ovr'
        ))
    ]),
    
    "Decision Tree": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            criterion='gini',
            max_features='sqrt'
        ))
    ]),
    
    "Random Forest": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1
        ))
    ]),
    
    "K-Nearest Neighbors": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(
            n_neighbors=7,
            weights='distance',
            algorithm='auto',
            leaf_size=30,
            metric='minkowski',
            p=2
        ))
    ]),
    
    "Support Vector Machine": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            probability=True,
            random_state=42,
            C=10.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced'
        ))
    ]),
    
    "XGBoost": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1
        ))
    ]),
    
    "Gradient Boosting": Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('classifier', GradientBoostingClassifier(
            random_state=42,
            n_estimators=150,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2
        ))
    ])
}

# -------------------------------
# Alternative: Models with GridSearchCV for hyperparameter tuning
# -------------------------------
def get_models_with_gridsearch():
    """
    Alternative approach using GridSearchCV for hyperparameter tuning
    Uncomment this section if you want to perform hyperparameter tuning
    """
    models_grid = {
        "Logistic Regression": {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, random_state=42))
            ]),
            'params': {
                'classifier__C': [0.1, 1.0, 10.0],
                'classifier__solver': ['liblinear', 'lbfgs'],
                'classifier__penalty': ['l2']
            }
        },
        
        "Random Forest": {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
            ]),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 15, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__max_features': ['sqrt', 'log2']
            }
        },
        
        "XGBoost": {
            'pipeline': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('classifier', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42))
            ]),
            'params': {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [3, 6, 9],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        }
    }
    return models_grid

# -------------------------------
# Train, Evaluate, Save Best Model
# -------------------------------
results = {}
detailed_results = {}
best_acc = 0
best_model_name = None
best_model = None

print("\n" + "="*60)
print("TRAINING AND EVALUATING MODELS")
print("="*60)

for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    print("-" * 40)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate accuracies
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing Accuracy: {test_acc:.4f}")
    print(f"Overfitting Check: {train_acc - test_acc:.4f}")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Store results
    results[name] = test_acc
    detailed_results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'overfitting': train_acc - test_acc,
        'model': model
    }
    
    # Check if this is the best model
    if test_acc > best_acc:
        best_acc = test_acc
        best_model = model
        best_model_name = name

# -------------------------------
# Save best model and results
# -------------------------------
best_model_path = os.path.join(models_dir, "model.pkl")
joblib.dump(best_model, best_model_path)

# Save detailed results
results_path = os.path.join(models_dir, "model_comparison_results.pkl")
joblib.dump(detailed_results, results_path)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"🏆 Best model: {best_model_name}")
print(f"🎯 Best accuracy: {best_acc:.4f}")
print(f"💾 Model saved at: {best_model_path}")

# Print summary table
print("\n📊 MODEL COMPARISON SUMMARY:")
print("-" * 70)
print(f"{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Overfitting':<12}")
print("-" * 70)
for name, details in detailed_results.items():
    print(f"{name:<25} {details['train_acc']:<12.4f} {details['test_acc']:<12.4f} {details['overfitting']:<12.4f}")

# -------------------------------
# Create and save visualizations
# -------------------------------
# 1. Accuracy comparison plot
plt.figure(figsize=(12, 8))
models_names = list(results.keys())
test_accuracies = [detailed_results[name]['test_acc'] for name in models_names]
train_accuracies = [detailed_results[name]['train_acc'] for name in models_names]

x = np.arange(len(models_names))
width = 0.35

plt.bar(x - width/2, train_accuracies, width, label='Training Accuracy', alpha=0.8, color='lightblue')
plt.bar(x + width/2, test_accuracies, width, label='Testing Accuracy', alpha=0.8, color='orange')

plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparison: Training vs Testing Accuracy')
plt.xticks(x, models_names, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

accuracy_plot_path = os.path.join(plots_dir, "classifier_accuracy_comparison.png")
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
plt.close()

# 2. Overfitting analysis plot
plt.figure(figsize=(10, 6))
overfitting_scores = [detailed_results[name]['overfitting'] for name in models_names]
colors = ['red' if score > 0.05 else 'green' for score in overfitting_scores]

plt.bar(models_names, overfitting_scores, color=colors, alpha=0.7)
plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold (0.05)')
plt.xlabel('Models')
plt.ylabel('Overfitting Score (Train Acc - Test Acc)')
plt.title('Overfitting Analysis by Model')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

overfitting_plot_path = os.path.join(plots_dir, "overfitting_analysis.png")
plt.savefig(overfitting_plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Accuracy comparison plot saved at: {accuracy_plot_path}")
print(f"✅ Overfitting analysis plot saved at: {overfitting_plot_path}")

# -------------------------------
# Feature importance for tree-based models
# -------------------------------
if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
    plt.figure(figsize=(10, 6))
    importance = best_model.named_steps['classifier'].feature_importances_
    feat_imp = pd.Series(importance, index=feature_cols).sort_values(ascending=True)
    
    feat_imp.plot(kind='barh', color='skyblue')
    plt.title(f'Feature Importance - {best_model_name}')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    feature_importance_path = os.path.join(plots_dir, "feature_importance.png")
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Feature importance plot saved at: {feature_importance_path}")

# -------------------------------
# Confusion Matrix for Best Model
# -------------------------------
print("\nGenerating Confusion Matrix for the best model...")
y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
# Get unique class labels from the data for the display labels
unique_labels = sorted(list(set(y)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
disp.plot(cmap=plt.cm.Blues, values_format='d', xticks_rotation='vertical')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()

cm_plot_path = os.path.join(plots_dir, "confusion_matrix.png")
plt.savefig(cm_plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Confusion Matrix plot saved at: {cm_plot_path}")

print("\n🎉 Training completed successfully!")
