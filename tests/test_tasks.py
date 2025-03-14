import pytest
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from tasks import (load_data, remove_outliers, scale_features,
                   encode_categorical_variables, prepare_regression_data,
                   build_regression_model, train_regression_model,
                   evaluate_regression_model, prepare_classification_data,
                   build_classification_model, train_classification_model,
                   evaluate_classification_model)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return load_data('winequality-red.csv', 'winequality-white.csv')


@pytest.fixture(scope="module")
def pre_process(data) -> pd.DataFrame:
    features = data.columns[:-2]
    return encode_categorical_variables(remove_outliers(data, features))


def test_load_data(data):
    assert not data.empty, "The data must not be empty."
    assert 'type' in data.columns, "The data must contain a 'type' column."
    assert data['type'].nunique() == 2, "There should be two types of wine."


def test_remove_outliers(data):
    features = data.drop(['quality', 'type'], axis=1)
    data_no_outliers = remove_outliers(data, features)
    assert data_no_outliers.shape[0] < data.shape[
        0], "The size of the data should decrease after removing outliers."


def test_encode_categorical_variables(data):
    data_encoded = encode_categorical_variables(data)
    assert data_encoded['type'].isin(
        [0, 1]).all(), "The 'type' column must be encoded as 0 and 1."


def test_scale_features(data):
    X = data.drop(['quality', 'type'], axis=1)
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    means = np.mean(X_train_scaled, axis=0)
    stds = np.std(X_train_scaled, axis=0)
    assert np.allclose(
        means, 0, atol=1e-7
    ), "The mean of the scaled features of the training set is not equal to 0."
    assert np.allclose(
        stds, 1, atol=1e-7
    ), "The standard deviation of the scaled features of the training set is not equal to 1."

    means_test = np.mean(X_test_scaled, axis=0)
    stds_test = np.std(X_test_scaled, axis=0)
    assert not np.allclose(
        means_test, 0, atol=1e-7
    ), "The mean of the scaled features of the test set is equal to 0."
    assert not np.allclose(
        stds_test, 1, atol=1e-7
    ), "The standard deviation of the scaled features of the test set is equal to 1."


def test_build_regression_model():
    input_dim = 12
    model = build_regression_model(input_dim)
    assert model.output_shape == (
        None, 1), "The output layer for regression must have dimension of 1."


def test_regression_workflow(pre_process):
    X_reg, y_reg = prepare_regression_data(pre_process)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42)
    X_train_scaled_reg, X_test_scaled_reg = scale_features(
        X_train_reg, X_test_reg)
    model = build_regression_model(X_train_scaled_reg.shape[1])
    history = train_regression_model(model,
                                     X_train_scaled_reg,
                                     y_train_reg,
                                     epochs=5)
    assert len(
        history.epoch
    ) == 5, "Model should be trained for specified ammount of epochs"
    regression_results = evaluate_regression_model(model, X_test_scaled_reg,
                                                   y_test_reg)
    assert regression_results['MSE'] > 0, "MSE must be greater than 0."
    assert -1 <= regression_results[
        'R2'] <= 1, "R² should be in the range of -1 to 1."


def test_build_classification_model():
    input_dim = 12
    num_classes = 3
    model = build_classification_model(input_dim, num_classes)
    assert model.output_shape == (
        None, num_classes
    ), "The output layer dimension must be equal to the number of classes."


def test_classification_workflow(pre_process):
    X_clf, y_clf = prepare_classification_data(pre_process)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42)
    X_train_scaled_clf, X_test_scaled_clf = scale_features(
        X_train_clf, X_test_clf)
    model = build_classification_model(X_train_scaled_clf.shape[1],
                                       y_clf.shape[1])
    history = train_classification_model(model,
                                         X_train_scaled_clf,
                                         y_train_clf,
                                         epochs=5)
    assert len(
        history.epoch
    ) == 5, "Model should be trained for specified ammount of epochs"
    classification_results = evaluate_classification_model(
        model, X_test_scaled_clf, y_test_clf)
    assert 0 <= classification_results[
        'accuracy'] <= 1, "Accuracy must be between 0 and 1."
    report = classification_results['report']
    assert isinstance(report, str) or isinstance(
        report,
        dict), "The classification report must be a string or dictionary."
