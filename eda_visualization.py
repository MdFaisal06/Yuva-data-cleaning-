import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
df = sns.load_dataset('titanic')

# 2. Descriptive Summary
print("--- SUMMARY STATISTICS ---")
print("Survival Rate:", f"{df['survived'].mean() * 100:.2f}%")
print(df[['age', 'fare', 'pclass']].describe())

# 3. Visualization 1: Survival by Gender
plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=df, palette=['#2b5c8f', '#d95f02'])
plt.title("Survival Probability by Gender", fontsize=11, fontweight='bold')
plt.xlabel("Gender")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig('survival_by_gender.png')
plt.close()

# 4. Visualization 2: Survival across Passenger Class & Gender
plt.figure(figsize=(7, 4))
sns.barplot(x='pclass', y='survived', hue='sex', data=df, palette=['#2b5c8f', '#d95f02'])
plt.title("Survival Rate across Class and Gender", fontsize=11, fontweight='bold')
plt.xlabel("Passenger Class (1st, 2nd, 3rd)")
plt.ylabel("Survival Rate")
plt.tight_layout()
plt.savefig('class_survival_breakdown.png')
plt.close()

# 5. Visualization 3: Correlation Matrix Heatmap
plt.figure(figsize=(7, 5))
numeric_cols = df[['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']].dropna()
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Numerical Feature Correlation Heatmap", fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.close()

print("EDA and Visualizations created successfully.")
