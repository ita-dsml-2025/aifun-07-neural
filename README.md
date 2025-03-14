# Neural Networks

## Description

In this task, you will learn to apply neural networks to the [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) dataset. This dataset consists of physicochemical properties of red and white wines, as well as quality ratings assigned by experts.

### Data Structure

| Characteristic                  | Variable Name              | Data Type      |
|---------------------------------|----------------------------|----------------|
| Fixed acidity                   | `fixed acidity`            | float          |
| Volatile acidity                | `volatile acidity`         | float          |
| Citric acid                     | `citric acid`              | float          |
| Residual sugar                  | `residual sugar`           | float          |
| Chlorides                       | `chlorides`                | float          |
| Free sulfur dioxide             | `free sulfur dioxide`      | float          |
| Total sulfur dioxide            | `total sulfur dioxide`     | float          |
| Density                         | `density`                  | float          |
| pH level                        | `pH`                       | float          |
| Sulphates                       | `sulphates`                | float          |
| Alcohol content                 | `alcohol`                  | float          |
| __Quality rating__              | `quality`                  | int            |
| __Type__                        | `type`                     | Categorical    |

- Fixed acidity: Concentration of non-volatile acids (e.g., tartaric acid) in g/dm³.
- Volatile acidity: Concentration of volatile acids (e.g., acetic acid) in g/dm³.
- Citric acid: Concentration of citric acid in g/dm³.
- Residual sugar: Amount of sugar remaining after fermentation in g/dm³.
- Chlorides: Concentration of chlorides (salt) in g/dm³.
- Free sulfur dioxide: Amount of free SO₂ in mg/dm³.
- Total sulfur dioxide: Total amount of SO₂ in mg/dm³.
- Density: Density of wine in g/cm³.
- pH level: Measure of acidity or alkalinity.
- Sulphates: Concentration of potassium sulphate in g/dm³.
- Alcohol content: Percentage of alcohol by volume (%).
- Quality rating (__target variable__): Wine quality rating (from 0 to 10) provided by expert wine tasters.
- Type: Indicates the type of wine (`'red'` or `'white'`). The type is not present in the data itself but is part of the filename.

## Preparation

Before completing the task, you need to set up the development environment.

### Creating a Python Virtual Environment

1. For Windows:
   ```powershell
   python -m venv venv
   ```

2. For Linux/macOS:
   ```bash
   python3 -m venv venv
   ```

### Activating the Virtual Environment

1. For Windows:
   ```powershell
   .\venv\Scripts\activate
   ```

2. For Linux/macOS:
   ```bash
   source venv/bin/activate
   ```

> After activating the virtual environment, you will see the prefix `(venv)` before the command line.

### Installing Required Packages from `requirements.txt`

1. For both systems:
   ```bash
   pip install -r requirements.txt
   ```

## Tasks

### Task 1: Data Preprocessing

- Loading data:
   - Implement the function `load_data(red_path, white_path)` to load the red and white wine datasets and combine them into one DataFrame.
      - The `type` characteristic (`'red'` or `'white'`) needs to be added to the dataset based on the filename from which the data is read.
      - You'll need to use pandas to read the CSV files and concatenate them.
      - Make sure to handle any potential missing values or inconsistencies between the two datasets.
- Data preprocessing:
   - Implement the function `remove_outliers(df, features)` to detect and process outliers in the dataset using the [IQR](https://medium.com/@pp1222001/outlier-detection-and-removal-using-the-iqr-method-6fab2954315d) method.
      - For each feature, calculate Q1 (25th percentile), Q3 (75th percentile), and IQR = Q3 - Q1.
      - Define lower bound = Q1 - 1.5 * IQR and upper bound = Q3 + 1.5 * IQR.
      - Remove rows where any feature value is outside these bounds.
      - Return the cleaned DataFrame.
   - Implement the function `scale_features(X_train, X_test)` to standardize the features.
      - Use `StandardScaler` from sklearn.preprocessing.
      - Fit the scaler on training data only.
      - Transform both training and test data.
      - Return the scaled data as numpy arrays.
   - If there are categorical variables in the dataset (e.g., `type`), encode them by implementing the function `encode_categorical_variables(data)`.
      - Use `LabelEncoder` from the `sklearn.preprocessing` module.
      - Apply the encoder to the 'type' column.
      - Return the DataFrame with encoded categorical variables.

### Task 2: Regression Using Neural Networks

- Data preparation:
   - Implement the function `prepare_regression_data(data)` to prepare the features and target variable.
      - Separate features (X) and target variable (y).
      - Features should include all columns except `quality` and `type`.
      - Target variable should be the `quality` column.
      - Return X and y as separate DataFrames/Series.
- Model development:
   - Implement the function `build_regression_model(input_dim)` to create a neural network regressor.
      - Create a Sequential model using `keras.Sequential()`.
      - Add at least 3 Dense layers with appropriate activation functions:
         - Input layer: Dense with `input_dim` neurons
         - Hidden layers: Dense with decreasing number of neurons (e.g., 64, 32, 16)
         - Output layer: Dense with 1 neuron (no activation function for regression)
      - Use appropriate activation functions (ReLU for hidden layers).
      - Compile the model with:
         - Loss function: 'mean_squared_error'
         - Optimizer: 'adam'
         - Metrics: ['mse']
- Model training:
   - Implement the function `train_regression_model(model, X_train, y_train)` to train the model.
      - Use `model.fit()` with the following parameters:
         - epochs=50
         - batch_size=32
         - validation_split=0.2
         - verbose=1
      - Return the training history.
- Model evaluation:
   - Implement the function `evaluate_regression_model(model, X_test, y_test)` to evaluate the model.
      - Use `model.predict()` to get predictions.
      - Calculate MSE using `mean_squared_error` from sklearn.metrics.
      - Calculate R² using `r2_score` from sklearn.metrics.
      - Return a dictionary with 'MSE' and 'R2' keys.

### Task 3: Classification Using Neural Networks

- Data preparation:
   - Convert quality ratings to categorical labels in the function `prepare_classification_data(data)`:
      - Define threshold values for wine categorization:
         - 0-5: 'Low' (class 0)
         - 6-7: 'Medium' (class 1)
         - 8-10: 'High' (class 2)
      - Create a new column with these categorical labels.
      - Separate features (X) and target variable (y).
      - Convert target variable to one-hot encoding using `keras.utils.to_categorical`.
      - Return X and y.
- Model development:
   - Implement the function `build_classification_model(input_dim, num_classes)` to create a neural network classifier.
      - Create a Sequential model using `keras.Sequential()`.
      - Add at least 3 Dense layers with appropriate activation functions:
         - Input layer: Dense with `input_dim` neurons
         - Hidden layers: Dense with decreasing number of neurons (e.g., 64, 32, 16)
         - Output layer: Dense with `num_classes` neurons
      - Use appropriate activation functions (ReLU for hidden layers, softmax for output).
      - Compile the model with:
         - Loss function: 'categorical_crossentropy'
         - Optimizer: 'adam'
         - Metrics: ['accuracy']
- Model training:
   - Implement the function `train_classification_model(model, X_train, y_train)` to train the model.
      - Use `model.fit()` with the following parameters:
         - epochs=50
         - batch_size=32
         - validation_split=0.2
         - verbose=1
      - Return the training history.
- Model evaluation:
   - Implement the function `evaluate_classification_model(model, X_test, y_test)` to evaluate the model.
      - Use `model.predict()` to get predictions.
      - Convert one-hot encoded predictions back to class labels.
      - Calculate accuracy using `accuracy_score` from sklearn.metrics.
      - Generate classification report using `classification_report` from sklearn.metrics.
      - Return a dictionary with 'accuracy' and 'report' keys.
