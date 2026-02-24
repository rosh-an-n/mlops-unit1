"""
data_stats.py - Load iris dataset and print basic stats
"""

import pandas as pd
from sklearn.datasets import load_iris

# load data
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(f"Shape: {df.shape}")
print(f"Features: {list(iris.feature_names)}")

# class counts
print("\nClass distribution:")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {(df['target'] == i).sum()}")

# basic stats
print("\nDescriptive stats:")
print(df.describe())

# missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# correlation matrix
print("\nCorrelation matrix:")
print(df.drop(columns=['target']).corr().round(2))

# min/max per feature
print("\nMin/Max per feature:")
for col in iris.feature_names:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}")
