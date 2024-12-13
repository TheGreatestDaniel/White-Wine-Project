# Data manipulation and analysis
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns

# Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

# Evaluation metrics
from sklearn.metrics import accuracy_score


# https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/winequality-white.csv
class generator():

    def __init__(self, dataset):
        self.url = dataset

        self.df = pd.read_csv(self.url)

        X = self.df.drop('quality', axis=1)
        y = self.df['quality']

        self.classifications = sorted(y.unique())

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the Decision Tree Classifier
        dt = DecisionTreeClassifier(random_state=42)

        # Define the parameter grid for tuning medium complexity
        param_grid = {
            # Criterion for the split (impurity measure)
            'criterion': ['gini', 'entropy'],
            # Depth of the tree, control overfitting
            'max_depth': list(range(1, 25)),
            # Minimum samples required to split a node
            'min_samples_split': [2, 10, 20],
            # Minimum samples required to be a leaf node
            'min_samples_leaf': [1, 5, 10],
            # Number of features to consider for splitting
            'max_features': [None, 'sqrt', 'log2']
        }

        # Set up GridSearchCV to find the best parameters
        grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

        # Fit GridSearchCV on the training data
        grid_search.fit(X_train, y_train)

        # Get the best decision tree model
        self.best_dt = grid_search.best_estimator_

        print(self.classifications)

    def getNode(self, node_index):
        tree = self.best_dt.tree_

        X = self.df.drop('quality', axis=1)

        # Node details
        left_child = tree.children_left[node_index]
        right_child = tree.children_right[node_index]

        feature_index = tree.feature[node_index]
        feature_name = "Leaf Node" if feature_index == -2 else X.columns[feature_index]

        threshold = tree.threshold[node_index]

        return node_index, feature_name, threshold, left_child, right_child

    def getLeft(self, index):
        tree = self.best_dt.tree_
        return tree.children_left[index]

    def getRight(self, index):
        tree = self.best_dt.tree_
        return tree.children_right[index]

