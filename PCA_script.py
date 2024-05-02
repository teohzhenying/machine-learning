import pandas as pd
from random import sample
import numpy as np
from numpy import linalg as LA

pizza = pd.read_csv('pizza.csv')

def train_test_split_function(dataset):
    """
    Used because target variable (y) has not yet been identified.
    :return: train set and test set (disregarding features and target)
    """
    length_of_dataset = len(dataset)
    train_set_length = round(length_of_dataset*0.8)

    # Generate a shuffled list of indices and split into train and test indices
    shuffled_indices = sample(range(length_of_dataset), length_of_dataset)
    train_indices = shuffled_indices[:train_set_length]
    test_indices = shuffled_indices[train_set_length:]

    # Create train and test sets
    train_set = dataset.iloc[train_indices]
    test_set = dataset.iloc[test_indices]
    
    return train_indices, test_indices, train_set, test_set

train_indices, test_indices, train_set, test_set = train_test_split_function(pizza)

# get the covariance matrix
train_set_length = len(train_set)
cov_matrix = np.zeros((5, 5))  

for x_idx in range(5):  
    for y_idx in range(5):
        x_diff = train_set.iloc[:, x_idx] - train_set.iloc[:, x_idx].mean()
        y_diff = train_set.iloc[:, y_idx] - train_set.iloc[:, y_idx].mean()
        cov_matrix[x_idx, y_idx] = (x_diff * y_diff).sum() / (train_set_length - 1)

print(cov_matrix)

eigenvalues, eigenvectors = LA.eig(cov_matrix)
print(eigenvalues)
print(eigenvectors)

# Sorting eigenvalues and eigenvectors
indices = eigenvalues.argsort()[::-1]  # This sorts the eigenvalues in descending order
sorted_eigenvalues = eigenvalues[indices]
sorted_eigenvectors = eigenvectors[:, indices]

# Keep the top 2 eigenvectors
top_2_eigenvectors = sorted_eigenvectors[:, :2]

# Centering the training data
train_set_centered = train_set - train_set.mean()

# Projecting the centered data onto the top 2 eigenvectors
x_projected = np.dot(train_set_centered, top_2_eigenvectors)
print(x_projected)

import matplotlib.pyplot as plt

# Assuming x_projected is the result from PCA projection
plt.figure(figsize=(8, 6))
plt.scatter(x_projected[:, 0], x_projected[:, 1], alpha=0.5)
plt.title('PCA Projection')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
