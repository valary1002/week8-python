# Iris Data Analysis Project

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# -----------------------------
# Task 1: Load and Explore Data
# -----------------------------

try:
    # Load dataset
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and structure
    print("\nData Types and Structure:")
    print(df.info())

    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Clean dataset (no missing values in Iris, but placeholder shown)
    # df.dropna(inplace=True) or df.fillna(value, inplace=True)
except Exception as e:
    print(f"Error loading dataset: {e}")

# --------------------------------
# Task 2: Basic Data Analysis
# --------------------------------

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Add species names
df['species'] = df['target'].map(dict(zip(range(3), iris.target_names)))

# Group by species and compute mean
grouped = df.groupby('species').mean()
print("\nMean of each feature by species:")
print(grouped)

# Interesting findings
print("\nObservations:")
print("- Setosa has significantly shorter petal lengths than other species.")
print("- Virginica tends to have the largest sepal and petal measurements.")

# --------------------------------
# Task 3: Data Visualization
# --------------------------------

# Set theme
sns.set(style="whitegrid")

# 1. Line Chart - Sepal Length over index
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal length (cm)'], label="Sepal Length", color='green')
plt.title("Line Chart of Sepal Length")
plt.xlabel("Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart - Average Petal Length by Species
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="species", y="petal length (cm)", palette="pastel")
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram - Distribution of Sepal Width
plt.figure(figsize=(6, 4))
plt.hist(df['sepal width (cm)'], bins=10, color='skyblue', edgecolor='black')
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# 4. Scatter Plot - Sepal Length vs Petal Length
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x="sepal length (cm)", y="petal length (cm)", hue="species", palette="deep")
plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()
