from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    X: Feature data.
    y: Target data.
    test_size: The proportion of the dataset to include in the test split, default is 0.2.
    random_state: Seed for random number generation, default is 42.

    Returns:
    X_train: Feature data for training.
    X_test: Feature data for testing.
    y_train: Target data for training.
    y_test: Target data for testing.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


# X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)
