import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib


# טעינת הנתונים
file_path = os.path.join('data', 'processed', 'Cleaned_Social_Media_Data.csv')
df = pd.read_csv(file_path)


# הכנת הנתונים
# קביעת משתנה המטרה
target_column = 'AcademicImpact' 

# חלוקה ל-X  ול-y
X = df.drop(columns=[target_column])
y = df[target_column]


# המרת משתנים קטגוריאליים למספרים עבור הפיצ'רים השונים
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X_encoded = pd.get_dummies(X, drop_first=True)

# חלוקה משולשת (Train 60%, Validation 20%, Test 20%)
X_train, X_temp, y_train, y_temp = train_test_split(X_encoded, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(f"Data split sizes -> Train: {len(X_train)} | Validation: {len(X_val)} | Test: {len(X_test)}")

# נרמול הנתונים
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# אימון והערכת מודלים על סט הוולידציה

# מודל 1:  רגרסיה לינארית
baseline_model = LinearRegression()
baseline_model.fit(X_train_scaled, y_train)
baseline_preds = baseline_model.predict(X_val_scaled)

baseline_rmse = np.sqrt(mean_squared_error(y_val, baseline_preds))
baseline_r2 = r2_score(y_val, baseline_preds)
print("\n---Linear Regression---")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"R-squared: {baseline_r2:.4f}")

# מודל 2: יער אקראי
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_val_scaled)

rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
rf_r2 = r2_score(y_val, rf_preds)
print("\n---Random Forest---")
print(f"RMSE: {rf_rmse:.2f}")
print(f"R-squared: {rf_r2:.4f}")


# מודל 3:KNN
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
knn_preds = knn_model.predict(X_val_scaled)

knn_rmse = np.sqrt(mean_squared_error(y_val, knn_preds))
knn_r2 = r2_score(y_val, knn_preds)
print("\n---KNN---")
print(f"RMSE: {knn_rmse:.2f}")
print(f"R-squared: {knn_r2:.4f}")


# בחירת המודל המנצח והרצה על סט המבחן הסופי
print("\n*** FINAL EVALUATION ON TEST SET ***")

final_preds = rf_model.predict(X_test_scaled)
final_rmse = np.sqrt(mean_squared_error(y_test, final_preds))
final_r2 = r2_score(y_test, final_preds)

print("-----------Final Model on Test Set (Random Forest)-----------") 
print(f"RMSE: {final_rmse:.2f}")
print(f"R-squared: {final_r2:.4f}")



# ניתוח חשיבות הפיצ'רים מתוך מודל ה-Random Fores
feature_importances = pd.DataFrame(
    rf_model.feature_importances_,
    index=X_train.columns,
    columns=['Importance']
).sort_values('Importance', ascending=False)

print("\n--- Top 10 Feature Importances (What affects grades the most?) ---")
print(feature_importances.head(10))


# שמירת המודל כקובץ כדי שנוכל להשתמש בו מתישהו בלי לאמן מחדש
model_filename = os.path.join('data', 'processed', 'rf_model.pkl')
joblib.dump(rf_model, model_filename)
print(f"\nModel saved successfully to: {model_filename}")


