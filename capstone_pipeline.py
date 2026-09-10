import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score

# 1. Ingestion & Preprocessing
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Pclass'], drop_first=True)

# 2. Unsupervised Clustering Integration
cluster_features = df[['Age', 'Fare']]
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(cluster_features)

# 3. Stratified Train-Test Split
feature_cols = ['Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Pclass_2', 'Pclass_3', 'Cluster']
X = df[feature_cols]
y = df['Survived']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 4. Supervised Predictive Model
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Final Evaluation
y_pred = model.predict(X_test)
print(f"Capstone Pipeline Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Export Predictions
output = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
output.to_csv('capstone_final_predictions.csv', index=False)
print("Capstone pipeline successfully completed.")
