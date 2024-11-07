import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


iris_data = datasets.load_iris()
iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris['species'] = iris_data.target


species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris['species'] = iris['species'].map(species_map)


plt.figure(figsize=(8, 6))
for species in iris['species'].unique():
    subset = iris[iris['species'] == species]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], label=species, alpha=0.7)
plt.title("Sepal Length vs Sepal Width by Species")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")
plt.show()


plt.figure(figsize=(8, 6))
species = iris['species'].unique()
petal_lengths = [iris[iris['species'] == s]['petal length (cm)'] for s in species]
plt.boxplot(petal_lengths, labels=species)
plt.title("Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()


features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
plt.figure(figsize=(12, 12))

for i, feature1 in enumerate(features):
    for j, feature2 in enumerate(features):
        plt.subplot(len(features), len(features), i * len(features) + j + 1)
        if i == j:
            plt.hist(iris[feature1], bins=10, alpha=0.7)
            plt.xlabel(feature1)
        else:
            for species in iris['species'].unique():
                subset = iris[iris['species'] == species]
                plt.scatter(subset[feature2], subset[feature1], label=species, alpha=0.5)
            if i == 0:
                plt.title(feature2)
            if j == 0:
                plt.ylabel(feature1)
plt.legend(title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



























































