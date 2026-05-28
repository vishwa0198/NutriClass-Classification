import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Load preprocessed (cleaned + normalized) data
df = pd.read_csv(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_clean.csv")

# Encode target column: Food_Name is the classification target per project requirements
target_col = 'Food_Name'
le = LabelEncoder()
df[target_col + '_encoded'] = le.fit_transform(df[target_col])

# Save encoded dataset
df.to_csv(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\data\processed\food_data_final.csv", index=False)

# Save label encoder
os.makedirs(r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models", exist_ok=True)
joblib.dump(le, r"C:\Guvi\NutriClass Food Classification Using Nutritional Data\models\label_encoder.pkl")
print("✅ Feature engineering completed.")
