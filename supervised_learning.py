import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Dataset Acquisition
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Preprocessing & Feature Engineering
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)

features = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_2', 'Pclass_3']
X = df[features]
y = df['Survived']

# 3. Stratified Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training: Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Cross-Validation & Test Set Evaluation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
y_pred = rf_model.predict(X_test)

print(f"5-Fold CV Mean Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Holdout Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save Model Predictions
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('model_predictions.csv', index=False)
print("Supervised learning pipeline executed successfully.")
