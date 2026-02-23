"""
data_stats.py - Load a dataset and print basic statistics.

This script uses the Iris dataset from sklearn as a sample ML dataset
and displays summary statistics including shape, feature names, 
class distribution, mean, standard deviation, and basic info.
"""

import numpy as np
from sklearn.datasets import load_iris
import pandas as pd


def load_data():
    """Load the Iris dataset and return it as a pandas DataFrame."""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df, iris.target_names


def print_basic_stats(df, target_names):
    """Print basic statistics of the dataset."""
    print("=" * 60)
    print("         IRIS DATASET - BASIC STATISTICS")
    print("=" * 60)

    # Shape
    print(f"\n📐 Dataset Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Feature names
    print(f"\n📋 Features: {list(df.columns[:-1])}")

    # Class distribution
    print("\n🏷️  Class Distribution:")
    for idx, name in enumerate(target_names):
        count = (df["target"] == idx).sum()
        print(f"   - {name}: {count} samples")

    # Descriptive statistics
    print("\n📊 Descriptive Statistics:")
    print(df.describe().to_string())

    # Missing values
    print(f"\n❓ Missing Values: {df.isnull().sum().sum()}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    df, target_names = load_data()
    print_basic_stats(df, target_names)
