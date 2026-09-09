import pandas as pd
import matplotlib.pyplot as plt

# Load a built-in dataset (no file needed)
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Quick look at the data
print(df.head())
print(df.describe())

# Plot sepal length by species
df.boxplot(column='sepal length (cm)', by='species')
plt.title('Sepal length by species')
plt.suptitle('')
plt.savefig('sepal_plot.png')
print("Plot saved as sepal_plot.png")

