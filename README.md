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
<img width="804" height="473" alt="image" src="https://github.com/user-attachments/assets/ca2bd40c-3bec-427d-9d9b-5434e3948c35" />
<img width="1004" height="575" alt="image" src="https://github.com/user-attachments/assets/3a2a4740-4766-4ff0-9c31-f3dd4aa8d071" />
<img width="644" height="551" alt="image" src="https://github.com/user-attachments/assets/c39062ef-60cc-46d1-b916-d8943617202c" />
<img width="602" height="671" alt="image" src="https://github.com/user-attachments/assets/88fc7513-a269-4b5b-81bc-73705a8e926a" />
<img width="604" height="676" alt="image" src="https://github.com/user-attachments/assets/05e0ce9a-5259-4274-a53f-9b86bc912831" />
<img width="607" height="676" alt="image" src="https://github.com/user-attachments/assets/26318269-9c75-4fc8-b6c0-d21ddce68d48" />





