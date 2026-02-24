import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(f"Dataset shape: {df.shape}")
print(f"Classes: {list(iris.target_names)}")

# split into train and test
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

# save model
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to models/model.pkl")
