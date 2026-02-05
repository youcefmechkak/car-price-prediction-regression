# Regression Models Comparison Project

## Overview
This project implements and compares several regression techniques to predict car prices using a real-world dataset. The goal is to study model performance, complexity, and generalization.

## Methods Implemented
- Linear Regression (baseline)
- Polynomial Regression (multiple degrees)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)

Both analytical solutions and gradient-based optimization are explored.

## Data Processing
- Removal of irrelevant columns
- Target encoding for categorical variables
- Feature scaling (Min-Max and Standard scaling)
- Exploratory data analysis using visualizations

## Evaluation Metrics
- Mean Squared Error (MSE)
- Coefficient of Determination (R²)
- Cross-validation for regularization parameters

## Results
Regularized models (Ridge/Lasso) demonstrate improved generalization compared to high-degree polynomial models, especially under limited data conditions.

## Repository Structure
- `src/`: main Python implementation
- `notebooks/`: exploratory analysis
- `data/`: dataset
- `results/`: generated figures

## Requirements
See `requirements.txt`

## Author
Mini-project for regression analysis course.
