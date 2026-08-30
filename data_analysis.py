
# עבודה עם טבלאות נתונים
import pandas as pd
import numpy as np
# איתור וקריאת קבצים מרובים מתיקיות המחשב
import glob 
import os

# הגדרת נתיב
file_path = os.path.join('data', 'raw', 'Social_Media_Impact_OnStudents.csv')

# טעינת הנתונים לתוך DataFrame
df = pd.read_csv(file_path, encoding='utf-8-sig')


# הצגת הממדים של הדאטה
print(f"Dataset shape: {df.shape}\n")

# הצגת מידע כללי על הנתונים - סוגי משתנים וערכים חסרים
print("\n--- Dataset Info ---")
df.info()

# הצגת 5 השורות הראשונות
print("\n--- First 5 rows ---")
print(df.head())

# הורדה של שורות עם ערכים חסרים
df_model = df.dropna()

# המרת משתנים טקסטואלים לפורמט אחיד (בלי אותיות קטנות וגדולות  והורדת רווחים מיותרים)
df['MostUsedPlatform'] = df['MostUsedPlatform'].str.strip().str.lower()

# הגדרת מילון להחלפה של המילים למספרים
severity_map = {
    'Not at all': 1,
    'Rarely': 2,
    'Slightly': 3,
    'Sometimes': 4,
    'Moderately': 5,
    'Often': 6,
    'Significantly': 7,
    'Always': 8
}

#בחירת הפיצ'רים שישתתפו בחישוב מדד הדחיינות
features_for_index = ['Procrastination', 'StudyDelay', 'CourseworkDistraction']

# יצירת עמודות מספריות זמניות לפי המילון שהגדרנו
for feature in features_for_index:
    # ניקוי רווחים מהטקסט    
    df[feature] = df[feature].str.strip()
    # יצירת עמודה חדשה שמכילה את הערך המספרי   
    df[f'{feature}_num'] = df[feature].map(severity_map)

# יצירת הפיצ'ר החדש: ממוצע של העמודות המספריות
# axis=1 אומר לפנדס לחשב את הממוצע על פני השורות (לכל סטודנט) ולא לאורך העמודה
df['Procrastination_Index'] = df[[f'{col}_num' for col in features_for_index]].mean(axis=1)

# הצגת התוצאה כדי לוודא שעבד כמו שצריך (מציגים רק את העמודות הרלוונטיות)
print("--- DataFrame with Procrastination Index ---")
print(df[['Procrastination', 'StudyDelay', 'Procrastination_Index']].head())

# הגדרת מילון שמפה כל ערך למספר הסידורי הרצוי מהקל אל הכבד
order_mapping = {
    'Not at all ': 1,
    'Slightly ' : 2,
    'Moderately ': 3,
    'Significantly ': 4,
}

# יצירת עמודת העזר החדשה שמקבלת את המספרים בהתאם לעמודת הפגיעה
df['ImpactOrder'] = df['ProductivityImpact'].map(order_mapping)


# ייצוא הדאטה הנקי לקובץ חדש כדי שיהיה מוכן ל-Power BI
df.to_csv('data/processed/Cleaned_Social_Media_Data.csv', index=False, encoding='utf-8-sig')







