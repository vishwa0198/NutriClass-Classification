import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
EVAL_DIR   = os.path.join(MODELS_DIR, "evaluation")
PLOTS_DIR  = os.path.join(MODELS_DIR, "plots")
DATA_PATH  = os.path.join(BASE_DIR, "data", "processed", "food_data_final.csv")

st.set_page_config(
    page_title="NutriClass — Food Classification",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# LOAD ASSETS
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(MODELS_DIR, "model.pkl"))

@st.cache_resource
def load_label_encoder():
    return joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

@st.cache_resource
def load_model_comparison():
    return joblib.load(os.path.join(MODELS_DIR, "model_comparison_results.pkl"))

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_eval_summary():
    with open(os.path.join(EVAL_DIR, "evaluation_summary.json")) as f:
        return json.load(f)

@st.cache_data
def load_classification_report():
    return pd.read_csv(os.path.join(EVAL_DIR, "classification_report.csv"), index_col=0)

@st.cache_data
def load_per_class():
    return pd.read_csv(os.path.join(EVAL_DIR, "per_class_performance.csv"))

@st.cache_data
def load_feature_importance():
    return pd.read_csv(os.path.join(EVAL_DIR, "feature_importance.csv"))

model    = load_model()
le       = load_label_encoder()
summary  = load_eval_summary()
df       = load_data()
per_cls  = load_per_class()
feat_imp = load_feature_importance()

try:
    model_comparison = load_model_comparison()
except Exception:
    model_comparison = None

FEATURE_COLS = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar',
                'Fiber', 'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/salad.png", width=80)
    st.title("NutriClass")
    st.caption("Food Classification Using Nutritional Data")
    st.divider()
    page = st.radio(
        "Navigate",
        ["🏠 Overview", "🔮 Predict Food", "📊 Model Performance",
         "📈 Visualizations", "🔍 Data Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Best Model:** XGBoost")
    st.metric("Test Accuracy", f"{summary['model_performance']['test_accuracy']*100:.2f}%")
    st.metric("Overfitting Score", f"{summary['model_performance']['overfitting_score']:.4f}")

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────
if page == "🏠 Overview":
    st.title("🥗 NutriClass: Food Classification")
    st.markdown(
        "Classifying food items into **10 categories** using nutritional attributes "
        "(calories, protein, fat, carbs, sugar, fiber, sodium, cholesterol, glycemic index, water content)."
    )
    st.divider()

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Samples", f"{len(df):,}")
    c2.metric("Food Classes", len(le.classes_))
    c3.metric("Best Accuracy", f"{summary['model_performance']['test_accuracy']*100:.2f}%")
    c4.metric("Misclassified", summary['error_analysis']['total_misclassified'])

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Class Distribution")
        class_counts = df['Food_Name'].value_counts().reset_index()
        class_counts.columns = ['Food', 'Count']
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = plt.cm.Set3(np.linspace(0, 1, len(class_counts)))
        bars = ax.barh(class_counts['Food'], class_counts['Count'], color=colors)
        ax.set_xlabel("Number of Samples")
        ax.set_title("Samples per Food Class")
        for bar, count in zip(bars, class_counts['Count']):
            ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                    str(count), va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Per-Class F1 Scores")
        fig, ax = plt.subplots(figsize=(6, 4))
        colors = ['#ff6b6b' if f == per_cls['F1-Score'].min() else
                  '#51cf66' if f == per_cls['F1-Score'].max() else '#74c0fc'
                  for f in per_cls['F1-Score']]
        bars = ax.barh(per_cls['Class'], per_cls['F1-Score'], color=colors)
        ax.set_xlim(0.97, 1.002)
        ax.set_xlabel("F1 Score")
        ax.set_title("F1 Score per Food Class")
        for bar, val in zip(bars, per_cls['F1-Score']):
            ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Misclassification Patterns")
    st.info(
        "**Burger → Pizza**: 14 times  |  **Pizza → Burger**: 12 times  |  "
        "**Ice Cream → Pizza**: 6 times  |  **Donut → Pizza**: 4 times  |  **Apple → Banana**: 2 times\n\n"
        "Pizza and Burger share similar macro-nutrient profiles, making them the most confused pair."
    )

# ─────────────────────────────────────────────
# PAGE: PREDICT
# ─────────────────────────────────────────────
elif page == "🔮 Predict Food":
    st.title("🔮 Predict Food from Nutrition")
    st.markdown("Enter the nutritional values below and the model will predict the food name.")
    st.divider()

    # Feature ranges from the (normalized) data
    raw_df = pd.read_csv(os.path.join(BASE_DIR, "data", "raw", "synthetic_food_dataset_imbalanced.csv"))

    col1, col2 = st.columns(2)
    inputs = {}

    feature_config = {
        'Calories':       ('kcal',  raw_df['Calories'].min(),       raw_df['Calories'].max(),       raw_df['Calories'].mean()),
        'Protein':        ('g',     raw_df['Protein'].min(),        raw_df['Protein'].max(),        raw_df['Protein'].mean()),
        'Fat':            ('g',     raw_df['Fat'].min(),            raw_df['Fat'].max(),            raw_df['Fat'].mean()),
        'Carbs':          ('g',     raw_df['Carbs'].min(),          raw_df['Carbs'].max(),          raw_df['Carbs'].mean()),
        'Sugar':          ('g',     raw_df['Sugar'].min(),          raw_df['Sugar'].max(),          raw_df['Sugar'].mean()),
        'Fiber':          ('g',     raw_df['Fiber'].min(),          raw_df['Fiber'].max(),          raw_df['Fiber'].mean()),
        'Sodium':         ('mg',    raw_df['Sodium'].min(),         raw_df['Sodium'].max(),         raw_df['Sodium'].mean()),
        'Cholesterol':    ('mg',    raw_df['Cholesterol'].min(),    raw_df['Cholesterol'].max(),    raw_df['Cholesterol'].mean()),
        'Glycemic_Index': ('',      raw_df['Glycemic_Index'].min(), raw_df['Glycemic_Index'].max(), raw_df['Glycemic_Index'].mean()),
        'Water_Content':  ('%',     raw_df['Water_Content'].min(),  raw_df['Water_Content'].max(),  raw_df['Water_Content'].mean()),
    }

    features_list = list(feature_config.items())
    for i, (feat, (unit, lo, hi, default)) in enumerate(features_list):
        col = col1 if i < 5 else col2
        label = f"{feat.replace('_', ' ')} ({unit})" if unit else feat.replace('_', ' ')
        inputs[feat] = col.number_input(label, min_value=float(lo), max_value=float(hi),
                                        value=float(round(default, 2)), step=0.01)

    st.divider()
    if st.button("🔍 Predict Food", type="primary", use_container_width=True):
        # Build input in original scale — the model uses the normalized data
        # We need to normalize the inputs the same way preprocessing did
        from sklearn.preprocessing import MinMaxScaler
        raw_num = raw_df[FEATURE_COLS]
        scaler = MinMaxScaler()
        scaler.fit(raw_num)

        raw_input = np.array([[inputs[f] for f in FEATURE_COLS]])
        scaled_input = scaler.transform(raw_input)

        pred_encoded = model.predict(scaled_input)[0]
        pred_proba   = model.predict_proba(scaled_input)[0]
        pred_name    = le.inverse_transform([pred_encoded])[0]
        confidence   = pred_proba[pred_encoded] * 100

        st.divider()
        r1, r2, r3 = st.columns([2, 1, 1])
        r1.success(f"### 🍽️ Predicted Food: **{pred_name}**")
        r2.metric("Confidence", f"{confidence:.1f}%")
        r3.metric("Prediction", "✅ High" if confidence >= 80 else "⚠️ Low")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame({
            'Food': le.classes_,
            'Probability': pred_proba * 100
        }).sort_values('Probability', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#ff6b6b' if f == pred_name else '#74c0fc' for f in prob_df['Food']]
        ax.barh(prob_df['Food'], prob_df['Probability'], color=colors)
        ax.set_xlabel("Probability (%)")
        ax.set_title("Class Probabilities")
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.divider()

    # Model comparison table
    if model_comparison:
        st.subheader("All Models — Comparison")
        rows = []
        for name, details in model_comparison.items():
            rows.append({
                'Model': name,
                'Train Accuracy': f"{details['train_acc']*100:.2f}%",
                'Test Accuracy':  f"{details['test_acc']*100:.2f}%",
                'CV Mean':        f"{details.get('cv_mean', 0)*100:.2f}%",
                'CV Std':         f"± {details.get('cv_std', 0)*100:.2f}%",
                'Overfitting':    f"{details['overfitting']:.4f}",
            })
        comp_df = pd.DataFrame(rows)
        # Highlight best row
        best_name = max(model_comparison, key=lambda n: model_comparison[n]['test_acc'])

        def highlight_best(row):
            return ['background-color: #d3f9d8' if row['Model'] == best_name else '' for _ in row]

        st.dataframe(
            comp_df.style.apply(highlight_best, axis=1),
            use_container_width=True, hide_index=True
        )

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        img_path = os.path.join(PLOTS_DIR, "classifier_accuracy_comparison.png")
        if os.path.exists(img_path):
            st.subheader("Training vs Testing Accuracy")
            st.image(img_path, use_container_width=True)

    with col2:
        img_path = os.path.join(PLOTS_DIR, "overfitting_analysis.png")
        if os.path.exists(img_path):
            st.subheader("Overfitting Analysis")
            st.image(img_path, use_container_width=True)

    st.divider()
    st.subheader("Per-Class Performance — Best Model (XGBoost)")
    st.dataframe(
        per_cls.style.background_gradient(subset=['Precision', 'Recall', 'F1-Score'], cmap='Greens'),
        use_container_width=True, hide_index=True
    )

# ─────────────────────────────────────────────
# PAGE: VISUALIZATIONS
# ─────────────────────────────────────────────
elif page == "📈 Visualizations":
    st.title("📈 Visualizations")
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Confusion Matrix", "ROC Curves", "Feature Importance",
        "Confidence Analysis", "Performance Metrics"
    ])

    with tab1:
        img = os.path.join(EVAL_DIR, "confusion_matrix.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

    with tab2:
        img = os.path.join(EVAL_DIR, "roc_curves.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
            roc = summary.get('roc_auc_scores', {})
            if roc:
                roc_df = pd.DataFrame(list(roc.items()), columns=['Class', 'AUC']).sort_values('AUC', ascending=False)
                st.dataframe(roc_df.style.background_gradient(subset=['AUC'], cmap='Greens'),
                             use_container_width=True, hide_index=True)

    with tab3:
        col1, col2 = st.columns([1, 1])
        with col1:
            img = os.path.join(EVAL_DIR, "feature_importance.png")
            if os.path.exists(img):
                st.image(img, use_container_width=True)
        with col2:
            st.subheader("Feature Importance Scores")
            st.dataframe(
                feat_imp.style.background_gradient(subset=['Importance'], cmap='Blues'),
                use_container_width=True, hide_index=True
            )

    with tab4:
        img = os.path.join(EVAL_DIR, "confidence_analysis.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)
        conf = summary.get('confidence_stats', {})
        if conf:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Avg Confidence",  f"{conf['mean_confidence']*100:.2f}%")
            c2.metric("Min Confidence",  f"{conf['min_confidence']*100:.2f}%")
            c3.metric("Max Confidence",  f"{conf['max_confidence']*100:.2f}%")
            c4.metric("Low Confidence Predictions", conf['low_confidence_count'])

    with tab5:
        img = os.path.join(EVAL_DIR, "performance_metrics.png")
        if os.path.exists(img):
            st.image(img, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ─────────────────────────────────────────────
elif page == "🔍 Data Explorer":
    st.title("🔍 Data Explorer")
    st.divider()

    # Filters
    col1, col2 = st.columns(2)
    selected_foods = col1.multiselect(
        "Filter by Food", options=sorted(df['Food_Name'].unique()),
        default=sorted(df['Food_Name'].unique())
    )
    selected_meal = col2.multiselect(
        "Filter by Meal Type", options=sorted(df['Meal_Type'].unique()),
        default=sorted(df['Meal_Type'].unique())
    )

    filtered = df[df['Food_Name'].isin(selected_foods) & df['Meal_Type'].isin(selected_meal)]
    st.caption(f"Showing {len(filtered):,} of {len(df):,} rows")
    st.dataframe(
        filtered[['Food_Name', 'Meal_Type', 'Preparation_Method', 'Is_Vegan',
                  'Is_Gluten_Free', 'Calories', 'Protein', 'Fat', 'Carbs', 'Sugar',
                  'Fiber', 'Sodium', 'Cholesterol', 'Glycemic_Index', 'Water_Content']
                 ].reset_index(drop=True),
        use_container_width=True, height=350
    )

    st.divider()
    st.subheader("Nutritional Statistics")
    st.dataframe(
        filtered[FEATURE_COLS].describe().T.style.background_gradient(cmap='Blues'),
        use_container_width=True
    )

    st.divider()
    st.subheader("Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    corr = filtered[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Pairwise Distribution by Food")
    feat = st.selectbox("Select feature to visualize", FEATURE_COLS)
    fig, ax = plt.subplots(figsize=(10, 4))
    for food in selected_foods:
        vals = filtered[filtered['Food_Name'] == food][feat]
        if len(vals) > 0:
            ax.hist(vals, bins=30, alpha=0.5, label=food, density=True)
    ax.set_xlabel(feat)
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of {feat} by Food Class")
    ax.legend(loc='upper right', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
