import pandas as pd
import numpy as np
import seaborn as sns

# Load dataset
df = sns.load_dataset('titanic')

# Impute missing values
df['age'] = df['age'].fillna(df['age'].median())
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
if 'deck' in df.columns:
    df.drop(columns=['deck'], inplace=True)

# Outlier capping using IQR
Q1 = df['fare'].quantile(0.25)
Q3 = df['fare'].quantile(0.75)
IQR = Q3 - Q1
df['fare'] = np.clip(df['fare'], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

# Categorical encoding & cleanup
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df.drop(columns=['embark_town', 'alive', 'class', 'who', 'adult_male'], inplace=True, errors='ignore')
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Save cleaned output
df.to_csv('cleaned_data.csv', index=False)
print("Data successfully cleaned and saved.")
