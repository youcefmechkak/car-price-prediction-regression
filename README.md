# Car Price Prediction Project

This project aims to develop a robust prediction system for car prices using various regression techniques. It explores data preprocessing, exploratory data analysis, and the comparison of different machine learning models to identify the most effective approach for price estimation.

## Project Structure
- `PredictRegres.py`: .
- `PredictRegres.ipynb`: The main script containing the initial exploration, visualization, the data processing and modeling logic.
- `car_price.csv`: The dataset used for training and testing.

## Methodology
1. **Preprocessing**: Handled unit conversion (Lakh/Crore), cleaned numerical strings, and managed categorical variables using One-Hot and Target Encoding.
2. **Exploration**: Visualized distributions and correlations to identify patterns and outliers.
3. **Modeling**: Implemented and compared the following:
   - Simple Linear Regression
   - Polynomial Regression (complexity analysis)
   - Regularized Models (Ridge and Lasso)
4. **Evaluation**: Models were assessed using Mean Squared Error (MSE) and Precision/R² scores via cross-validation.

## Installation
Clone the repository and install the required packages:
```bash
pip install -r requirements.txt
