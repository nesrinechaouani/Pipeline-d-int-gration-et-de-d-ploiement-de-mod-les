
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import mlflow
import mlflow.sklearn

# Étape 1 : Exploration et Préparation des Données
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    print("Aperçu des données :\n", df.head())
    print("\nRésumé statistique :\n", df.describe())
    print("\nValeurs manquantes :\n", df.isnull().sum())
    df.fillna(df.mean(), inplace=True)
    sns.histplot(df['mean_temp'], kde=True)
    plt.title("Distribution de la température moyenne")
    plt.show()
    return df

# Étape 2 : Construction et Entraînement du Modèle
def train_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }
    param_grids = {
        "Ridge": {"alpha": [0.1, 1.0, 10.0]},
        "Lasso": {"alpha": [0.01, 0.1, 1.0]},
        "RandomForest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]}
    }
    experiment_results = []
    mlflow.set_experiment("London Temperature Prediction")
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            if model_name in param_grids:
                grid = GridSearchCV(model, param_grids[model_name], scoring="neg_root_mean_squared_error", cv=3)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                model.fit(X_train, y_train)
                best_model = model
                best_params = {}
            y_pred = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mlflow.log_params(best_params)
            mlflow.log_metric("rmse", rmse)
            mlflow.sklearn.log_model(best_model, artifact_path=model_name)
            experiment_results.append({"model": model_name, "rmse": rmse, "best_params": best_params})
            print(f"Modèle {model_name}: RMSE = {rmse:.2f}")
    return pd.DataFrame(experiment_results)

# Étape 3 : Pipeline Principal
def main():
    data_file = "2025_Case+Study_MLOps_Data.csv"
    df = load_and_prepare_data(data_file)
    target = "mean_temp"
    features = df.drop(columns=[target]).select_dtypes(include=[np.number]).columns.tolist()
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    experiment_results = train_models(X_train, X_test, y_train, y_test)
    print("\nRésultats des expérimentations :\n", experiment_results)
    experiment_results.to_csv("experiment_results.csv", index=False)

if __name__ == "__main__":
    main()
