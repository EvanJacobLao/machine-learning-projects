from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Load and assign headers to data

iris = pd.read_csv(Path(__file__).parent / "iris.csv", header=None)

species_type = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica" : 2
}

iris.columns = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width",
    "Species"
]

# replace value assigned to column "Species" to a number

iris["Species"] = iris["Species"].replace(species_type).astype(int)

# euclidean_distance computation

def euclidean_distance(vector_a, vector_b):
    distance = 0.0
    for index in range(len(vector_a)):
        distance += np.square(vector_a[index] - vector_b[index])
    return np.sqrt(distance)

# Split data set - 50% train, 50% test

train_df, test_df = train_test_split(
    iris,
    test_size=0.5,
    random_state=1,
    shuffle=True
)

# Extract values of features and species

features = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width",]
x_train = train_df[features].values
y_train = train_df["Species"].values

x_test = test_df[features].values
y_test = test_df["Species"].values


def nearest_neighbor(x_train, sample_comparison, k):
    """
    Compare each feature value in x_train to sample_comparison
    List out the top k species that are nearest
    """
    distances = []
    for index in range(len(x_train)):
        dist = euclidean_distance(x_train[index], sample_comparison)
        distances.append(dist)
    return np.argsort(distances)[:k]

# Sample Test

sample_comparison = x_test[0]
k = 5
neighbor_indexes = nearest_neighbor(x_train, sample_comparison, k)
print(f"Nearest {k} neighbors found for are {x_test[0]}: \n")
for index in neighbor_indexes:
    print(x_train[index])

def count_vote(neighbor_indexes, y_train):
    """
    Count how many of each species is returned from nearest neighbor
    """
    voting_results = {}
    for index in neighbor_indexes:
        species = y_train[index]
        voting_results[species] = voting_results.get(species, 0) + 1
    return max(voting_results, key=voting_results.get)

# Test Prediction and Compare with Actual

predicted_species = count_vote(neighbor_indexes, y_train)
print(f"Predicted species: {predicted_species}")
print(f"Actual species: {y_test[0]}")

# Create reusable class

class MyKNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.x_train = None
        self.y_train = None

    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        predictions = []

        # For each flower in x_test predict species
        for sample_comparison in x_test:
            neighbor_indexes = nearest_neighbor(
                self.x_train, sample_comparison, self.k
            )
            predicted_species = count_vote(
                neighbor_indexes, self.y_train
            )
            predictions.append(predicted_species)

        return np.array(predictions)


# Test Class

model = MyKNNClassifier(k=3)
model.fit(x_train, y_train)
predictions = model.predict(x_test)

# accuracy_score compares both arrays for each prediction/actual species
accuracy = accuracy_score(y_test, predictions) * 100
error = 100 - accuracy

print(f"Accuracy: {accuracy:.2f}%")
print(f"Error rate: {error:.2f}%")