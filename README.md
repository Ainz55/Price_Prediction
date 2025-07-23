# Taxi Trip Price Prediction

This project implements three machine learning models to predict taxi trip prices: K-Nearest Neighbors (KNN), Decision Tree, and an ensemble of these models. The project includes data loading, preprocessing, model training, performance evaluation, and result visualization.

## 📌 Project Structure

1. **Data Loading and Preprocessing**:
   - Data is loaded from `taxi_trip_pricing.csv`.
   - Missing values are removed.
   - Categorical features are encoded into numerical values.

2. **Models**:
   - **KNN (K-Nearest Neighbors)**: A regression algorithm based on distance metrics.
   - **Decision Tree**: A regression algorithm using MSE to determine the best splits.
   - **Ensemble**: Combines predictions from KNN and Decision Tree by averaging.

3. **Metric**:
   - Mean Absolute Error (MAE) is used to evaluate model performance.

4. **Visualization**:
   - Distribution of the target variable (trip price).
   - Comparison of model predictions on test data.
   - Scatter plots for each model's predictions vs. true values.
   - MAE comparison across models.

## 📊 Results

After training and testing the models, the MAE values are displayed:
- **KNN**: [MAE value]
- **Decision Tree**: [MAE value]
- **Ensemble**: [MAE value]

## 📈 Plots

1. **Trip Price Distribution**:
   - A histogram showing the distribution of the target variable.

2. **Prediction Comparison**:
   - A plot comparing true and predicted values for the first 50 test points.

3. **Scatter Plots**:
   - For KNN, Decision Tree, and the ensemble, showing the relationship between true and predicted values.

4. **MAE Comparison**:
   - A bar chart displaying the MAE for each model.

## 🔧 Dependencies

The project requires the following libraries:
- `pandas`
- `numpy`
- `matplotlib`

Install them using:
```
pip install pandas numpy matplotlib
```
____
## Output
``📊 MAE:
1. KNN:      11.795061061946901
2. Tree:     12.439291466899261
3. Ensemble: 10.30312621920808
``
____
## Visualization
