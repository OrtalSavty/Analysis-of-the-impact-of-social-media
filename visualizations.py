# -*- coding: utf-8 -*-

import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# טעינת הנתונים הנקיים
data_path = os.path.join('data', 'processed', 'Cleaned_Social_Media_Data.csv')
df = pd.read_csv(data_path)

# טעינת מודל היער האקראי השמור
model_path = os.path.join('data', 'processed', 'rf_model.pkl')
rf_model = joblib.load(model_path)


# יצירת גרפים

# גרף 1: איזה פיצ'ר הכי משפיע
# שחזור שמות העמודות בדיוק כפי שהמודל ראה אותן בשלב האימון
X = df.drop(columns=['AcademicImpact'])
X_encoded = pd.get_dummies(X, drop_first=True)

# יצירת טבלת חשיבות הפיצ'רים מתוך המודל
feature_importances = pd.DataFrame(
    rf_model.feature_importances_,
    index=X_encoded.columns,
    columns=['Importance']
).sort_values('Importance', ascending=False)

# הגדרת סגנון הגרף
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# יצירת גרף עמודות אופקי מתוך 10 הפיצ'רים המובילים
sns.barplot(
    x=feature_importances['Importance'].head(10), 
    y=feature_importances.head(10).index, 
    hue=feature_importances.head(10).index,
    palette="viridis",
    legend=False
)

# עיצוב הכותרות
plt.title('Top 10 Features Impacting Academic Performance', fontsize=16, fontweight='bold')
plt.xlabel('Importance Weight (Influence)', fontsize=12)
plt.ylabel('Student Behavior / Feature', fontsize=12)

# הצגת הגרף
plt.tight_layout()
plt.show()



# גרף 2: השפעת שעות גלישה על פגיעה אקדמית

# סינון הנתונים - משאירים רק נשים וגברים
df['Gender'] = df['Gender'].astype(str).str.strip()

# סינון בטוח רק לנשים וגברים
df_filtered = df[df['Gender'].isin(['Female', 'Male', 'female', 'male'])]

custom_colors = {"Female": "hotpink", "Male": "dodgerblue", "female": "hotpink", "male": "dodgerblue"}

plt.figure(figsize=(12, 6))
sns.violinplot(
    data=df_filtered, 
    x='SocialMediaHours', 
    y='Procrastination_Index', 
    hue='Gender',             
    split=True,
    palette=custom_colors
)

plt.title('Impact of Social Media Hours on Procrastination by Gender', fontsize=16, fontweight='bold')
plt.xlabel('Daily Social Media Hours', fontsize=12)
plt.ylabel('Procrastination Index', fontsize=12)
plt.xticks(rotation=45) 
plt.tight_layout()
plt.show()



# גרף 3: מטריצת קורלציה למשתנים מספריים
plt.figure(figsize=(10, 8))

# סינון ה-DataFrame רק לעמודות מספריות
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# חישוב הקורלציה (מקדמי פירסון)
correlation_matrix = numeric_df.corr()

# יצירת ה-Heatmap
sns.heatmap(
    correlation_matrix, 
    annot=True,          
    cmap='coolwarm',   
    fmt=".2f",         
    linewidths=.5
)

plt.title('Correlation Heatmap of Numeric Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()