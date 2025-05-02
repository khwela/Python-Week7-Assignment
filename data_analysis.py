# Import  libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
# Load the Iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Check the structure of the dataset
print("\nDataset Info:")
print(df.info()) 

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Perform groupings and compute mean
grouped = df.groupby('target').mean()
print("\nMean of numerical columns grouped by target:")
print(grouped)

# Task 3: Data Visualization
# Line Chart
plt.figure(figsize=(8, 5))
df.iloc[:, :-1].mean().plot(kind='line', marker='o')
plt.title('Average Measurements of Iris Features')
plt.xlabel('Features')
plt.ylabel('Average Value')
plt.grid()
plt.show()

# Bar Chart
plt.figure(figsize=(8, 5))
grouped['sepal length (cm)'].plot(kind='bar', color='skyblue')
plt.title('Average Sepal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Sepal Length (cm)')
plt.xticks(ticks=[0, 1, 2], labels=iris.target_names, rotation=0)
plt.show()

# Histogram
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal length (cm)'], kde=True, bins=15, color='green')
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='target', data=df, palette='viridis')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', labels=iris.target_names)
plt.show()