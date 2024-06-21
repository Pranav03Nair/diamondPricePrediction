# Diamonds Price Prediction

## Overview
This project aims to analyze the characteristics of diamonds and build a predictive model to estimate their prices using various features. The dataset used for this analysis is `diamonds.csv`, which contains information about diamond attributes and their respective prices.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Description](#data-description)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Preprocessing](#preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, you need to have Python installed on your system along with the required libraries. You can install the necessary libraries using the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Data Description
The dataset contains the following columns:

- carat: Carat weight of the diamond
- cut: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- color: Diamond color, from J (worst) to D (best)
- clarity: A measurement of how clear the diamond is (I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF)
- depth: Total depth percentage
- table: Width of the top of the diamond relative to the widest point
- price: Price of the diamond
- x: Length in mm
- y: Width in mm
- z: Depth in mm

## Exploratory Data Analysis
We performed various visualizations to understand the distribution and relationships between the features:

- Histogram of diamond prices
- Kernel density estimate of diamond prices
- Cumulative distribution of diamond prices
- Scatter plot of carat weight vs. diamond price
- Box plots of carat weight and price
- Pair plot of selected features
- Radial histogram of diamond prices
- 3D scatter plot and surface plot of carat, depth, and price

## Preprocessing
The preprocessing steps included:

- Encoding categorical features (cut, color, clarity) using LabelEncoder
- Scaling numerical features using StandardScaler

## Modeling
We used a DecisionTreeRegressor to build the predictive model:

```
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Split the data
X = data_df.drop(['price'], axis=1)
y = data_df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Encode categorical features
label_encoder = LabelEncoder()
X_train['cut'] = label_encoder.fit_transform(X_train['cut'])
X_test['cut'] = label_encoder.transform(X_test['cut'])
X_train['color'] = label_encoder.fit_transform(X_train['color'])
X_test['color'] = label_encoder.transform(X_test['color'])
X_train['clarity'] = label_encoder.fit_transform(X_train['clarity'])
X_test['clarity'] = label_encoder.transform(X_test['clarity'])

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = tree_model.predict(X_test_scaled)

```

## Evaluation
The model's performance was evaluated using the following metrics:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R2 Score

```
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'Mean Absolute Error: {mae:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')
print(f'R2 Score: {r2:.2f}')
```

## Results
- Mean Squared Error: 295153.11
- Mean Absolute Error: 292.31
- Root Mean Squared Error: 542.98
- R2 Score: 0.97

## Contributing
If you would like to contribute to this project, please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.