# Predictive Maintenance Model

## Overview

This project focuses on developing a predictive maintenance model using machine learning techniques. The goal is to predict equipment failures based on sensor data from a manufacturing process. The dataset includes various sensor readings and failure indicators to build a model that can anticipate equipment failures before they occur.

## Dataset

The dataset contains the following columns:

- **UDI**: Unique identifier for each record
- **Product ID**: Identifier for the product
- **Type**: Type of the product (M or L)
- **Air temperature [K]**: Air temperature in Kelvin
- **Process temperature [K]**: Process temperature in Kelvin
- **Rotational speed [rpm]**: Rotational speed in RPM
- **Torque [Nm]**: Torque in Newton-meters
- **Tool wear [min]**: Tool wear in minutes
- **Target**: Target variable (0 for no failure, 1 for failure)
- **Failure Type**: Type of failure (if any)

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/predictive-maintenance-model.git
   cd predictive-maintenance-model
   ```

2. **Install required libraries:**

   You can install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` should include:

   ```
   pandas
   numpy
   scikit-learn
   matplotlib
   seaborn
   ```

## Usage

1. **Load the dataset:**

   ```python
   import pandas as pd

   # Load the dataset
   data = pd.read_csv('path/to/your_dataset.csv')
   ```

2. **Data Preprocessing:**

   ```python
   from sklearn.preprocessing import LabelEncoder, StandardScaler

   # Encode categorical variables
   label_encoder = LabelEncoder()
   data['Product ID'] = label_encoder.fit_transform(data['Product ID'])
   data['Type'] = label_encoder.fit_transform(data['Type'])

   # Separate features and target
   X = data.drop(['UDI', 'Target', 'Failure Type'], axis=1)
   y = data['Failure Type'].apply(lambda x: 1 if x == 'No Failure' else 0)

   # Normalize features
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Train-Test Split:**

   ```python
   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
   ```

4. **Model Training and Evaluation:**

   ```python
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import classification_report, roc_auc_score

   # Initialize and train the model
   model = RandomForestClassifier(random_state=42)
   model.fit(X_train, y_train)

   # Make predictions
   y_pred = model.predict(X_test)

   # Evaluate the model
   print("Classification Report:\n", classification_report(y_test, y_pred))
   print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))
   ```

5. **Hyperparameter Tuning:**

   ```python
   from sklearn.model_selection import GridSearchCV

   # Define parameter grid
   param_grid = {
       'n_estimators': [100, 200, 300],
       'max_depth': [10, 20, None]
   }

   # Initialize GridSearchCV
   grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
   grid_search.fit(X_train, y_train)

   # Best model
   best_model = grid_search.best_estimator_

   # Evaluate the best model
   y_pred_best = best_model.predict(X_test)
   print("Classification Report:\n", classification_report(y_test, y_pred_best))
   print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_best))
   ```

## Results

- **Classification Report**: Provides precision, recall, and F1-score metrics for each class.
- **ROC-AUC Score**: Measures the model's ability to distinguish between the classes.

## Contributing

Feel free to fork the repository and submit pull requests for improvements. If you find any issues, please open an issue in the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms and tools.
- [pandas](https://pandas.pydata.org/) for data manipulation and analysis.
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for data visualization.