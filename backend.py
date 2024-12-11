import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


def build_decision_tree(data, target_column, test_size=0.2, random_state=42):
    """
    Generalized function to build and evaluate a Decision Tree model.

    Parameters:
        data (pd.DataFrame): The dataset as a pandas DataFrame.
        target_column (str): The name of the target column in the dataset.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the best parameters, test accuracy, and the trained model.
    """
    # Split features and target
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the Decision Tree Classifier
    dt = DecisionTreeClassifier(random_state=random_state)

    # Define the parameter grid for tuning
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': list(range(1, 25)),
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': [None, 'sqrt', 'log2']
    }

    # Set up GridSearchCV to find the best parameters
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    # Fit GridSearchCV on the training data
    grid_search.fit(X_train, y_train)

    # Get the best decision tree model
    best_dt = grid_search.best_estimator_

    # Evaluate the model on the test set
    y_pred = best_dt.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    return {
        'best_params': grid_search.best_params_,
        'test_accuracy': test_accuracy,
        'model': best_dt
    }


# Example usage (remove or modify when integrating into a larger system):
if __name__ == "__main__":
    # Load dataset (replace with any dataset file path or DataFrame)
    url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/winequality-white.csv'
    dataset = pd.read_csv(url)

    # Target column in the dataset
    target = 'quality'

    # Build and evaluate the decision tree
    result = build_decision_tree(data=dataset, target_column=target)

    # Output the results
    print(f"Best Parameters: {result['best_params']}")
    print(f"Test Accuracy: {result['test_accuracy']:.2f}")
