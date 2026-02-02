# NutriClass: Food Classification Using Nutritional Data

## Project Overview
NutriClass is a machine learning project designed to classify food items into specific categories (or exact food names) based on their nutritional attributes such as calories, proteins, carbohydrates, fats, and sugar. This system aims to support smart dietary applications, health monitoring tools, and meal planning platforms by providing accurate food identification from nutritional profiles.

## Project Structure
```
├── data/
│   ├── raw/                # Original dataset (synthetic_food_dataset_imbalanced.csv)
│   └── processed/          # Cleaned and encoded data
├── models/                 # Saved models and plots
│   └── plots/              # Visualization of results
├── src/
│   ├── data_preprocessing.py   # Handles missing values, duplicates, outliers, normalization
│   ├── feature_engineering.py  # Label encoding and feature preparation
│   └── train_models.py         # Model training, evaluation, and saving
└── README.md
```

## Requirements Verification
This project has been validated against the following requirements:

| Requirement | Status | Implementation Details |
| :--- | :--- | :--- |
| **Data Preprocessing** | ✅ | Handled missing values, duplicates, outliers, and normalization in `data_preprocessing.py`. |
| **Feature Engineering** | ✅ | Implemented label encoding in `feature_engineering.py`. |
| **Model Selection** | ✅ | Trained and compared Logistic Regression, Decision Tree, Random Forest, KNN, SVM, XGBoost, and Gradient Boosting. |
| **Evaluation Metrics** | ✅ | Evaluated using Accuracy, Precision, Recall, F1-score, and Confusion Matrix. |
| **Business Use Case** | ✅ | Models predict food class/name from nutritional metrics, enabling diet plan adherence. |

## Feature Engineering Note
- **Label Encoding**: The target variable `Meal_Type` is encoded using `LabelEncoder`.
- **Feature Selection**: Tree-based models (Random Forest, XGBoost) provide feature importance scores, which are visualized in the `models/plots/` directory.

## Model Performance Considerations
The models show varying degrees of accuracy, with **Support Vector Machine (SVM)** currently performing best on the test set. Note that the low overall accuracy (~26%) suggests the synthetic dataset implies significant overlap between classes or high noise, which is common in synthetic data. Real-world data may yield different results.
- **Overfitting**: Random Forest and KNN showed signs of overfitting (high train accuracy, low test accuracy).
- **Best Generalization**: SVM and Gradient Boosting showed more stable generalization gaps.

## Getting Started

### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`

### Installation
```bash
pip install pandas numpy scikit-learn xgboost matplotlib joblib
```

### Running the Pipeline

1.  **Preprocess Data**:
    Clean the raw data and normalize features.
    ```bash
    python src/data_preprocessing.py
    ```

2.  **Feature Engineering**:
    Encode the target labels.
    ```bash
    python src/feature_engineering.py
    ```

3.  **Train and Evaluate Models**:
    Train all classifiers, generate metrics, and save the best model.
    ```bash
    python src/train_models.py
    ```

## Results
After running the pipeline, you can find:
- **Best Model**: Saved as `models/model.pkl`
- **Visualizations**: Located in `models/plots/`
    - `classifier_accuracy_comparison.png`
    - `overfitting_analysis.png`
    - `confusion_matrix.png`
    - `feature_importance.png`

## Future Improvements
- **Hyperparameter Tuning**: Uncomment the `get_models_with_gridsearch` function in `train_models.py` to run exhaustive search for better parameters.
- **Data Augmentation**: Collect more real-world samples to improve class separability.
- **Deep Learning**: Explore Neural Networks for potentially better pattern recognition in high-dimensional nutritional data.
