import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# --------------------------------------------------
# 2. Preprocessing
# --------------------------------------------------
def preprocess_data(df, target_column="price"):
    # Drop duplicates
    df = df.drop_duplicates()

    # Separate target
    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Handle categorical variables (target encoding / mean encoding)
    for col in X.select_dtypes(include=["object"]).columns:
        mean_encoding = df.groupby(col)[target_column].mean()
        X[col] = X[col].map(mean_encoding)

    # Handle missing values
    X = X.fillna(X.mean())

    return X, y


# --------------------------------------------------
# 3. Train-test split & scaling
# --------------------------------------------------
def split_and_scale(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


# --------------------------------------------------
# 4. Linear Regression
# --------------------------------------------------
def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# --------------------------------------------------
# 5. Polynomial Regression
# --------------------------------------------------
def polynomial_regression(X_train, X_test, y_train, y_test, degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# --------------------------------------------------
# 6. Ridge Regression
# --------------------------------------------------
def ridge_regression(X_train, X_test, y_train, y_test, alpha):
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


# --------------------------------------------------
# 7. Lasso Regression with CV
# --------------------------------------------------
def lasso_regression_cv(X, y, alpha_list):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_alpha = None
    best_mse = np.inf

    for alpha in alpha_list:
        mse_list = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = Lasso(alpha=alpha, max_iter=10000)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            mse_list.append(mean_squared_error(y_test, y_pred))

        avg_mse = np.mean(mse_list)

        if avg_mse < best_mse:
            best_mse = avg_mse
            best_alpha = alpha

    return best_alpha, best_mse


# --------------------------------------------------
# 8. Main
# --------------------------------------------------
def main():
    # Path to dataset
    data_path = "data/car_price.csv"

    df = load_data(data_path)
    X, y = preprocess_data(df, target_column="price")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)

    print("===== Linear Regression =====")
    mse, r2 = linear_regression(X_train, X_test, y_train, y_test)
    print(f"MSE: {mse:.2f}, R2: {r2:.4f}")

    print("\n===== Polynomial Regression =====")
    for degree in [2, 3, 4]:
        mse, r2 = polynomial_regression(
            X_train, X_test, y_train, y_test, degree
        )
        print(f"Degree {degree} -> MSE: {mse:.2f}, R2: {r2:.4f}")

    print("\n===== Ridge Regression =====")
    for alpha in [0.1, 1.0, 10.0]:
        mse, r2 = ridge_regression(
            X_train, X_test, y_train, y_test, alpha
        )
        print(f"Alpha {alpha} -> MSE: {mse:.2f}, R2: {r2:.4f}")

    print("\n===== Lasso Regression (CV) =====")
    alpha_list = [0.001, 0.01, 0.1, 1.0]
    best_alpha, best_mse = lasso_regression_cv(X_train, y_train, alpha_list)
    print(f"Best alpha: {best_alpha}, CV MSE: {best_mse:.2f}")


if __name__ == "__main__":
    main()
