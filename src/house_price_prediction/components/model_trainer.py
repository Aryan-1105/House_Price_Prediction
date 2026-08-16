import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.model_selection import (
    KFold,
    cross_validate,
    RandomizedSearchCV,
)

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging
from src.house_price_prediction.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

    leaderboard_file_path: str = os.path.join("artifacts", "model_comparison.csv")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Starting model training")

            # =========================================================
            # 1. Split features and target
            # =========================================================

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(f"X_train shape: {X_train.shape}")

            logging.info(f"X_test shape: {X_test.shape}")

            # =========================================================
            # 2. Define baseline models
            # =========================================================

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=0.001, max_iter=10000),
                "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(
                    n_estimators=300, random_state=42, n_jobs=-1
                ),
                "Extra Trees": ExtraTreesRegressor(
                    n_estimators=300, random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                ),
                "CatBoost": CatBoostRegressor(
                    iterations=300,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="RMSE",
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                ),
            }

            # =========================================================
            # 3. 5-Fold Cross Validation
            # =========================================================

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
            }

            results = []

            print("\n")
            print("=" * 95)
            print("MODEL COMPARISON - 5 FOLD CROSS VALIDATION")
            print("=" * 95)

            for model_name, model in models.items():

                print(f"\nEvaluating: {model_name}")

                logging.info(f"Evaluating {model_name}")

                cv_results = cross_validate(
                    estimator=model,
                    X=X_train,
                    y=y_train,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=-1,
                    return_train_score=True,
                    error_score="raise",
                )

                mean_cv_r2 = cv_results["test_r2"].mean()

                std_cv_r2 = cv_results["test_r2"].std()

                mean_cv_mae = -cv_results["test_mae"].mean()

                mean_cv_rmse = np.sqrt(-cv_results["test_mse"].mean())

                mean_train_r2 = cv_results["train_r2"].mean()

                results.append(
                    {
                        "Model": model_name,
                        "CV R2": mean_cv_r2,
                        "CV R2 Std": std_cv_r2,
                        "CV MAE": mean_cv_mae,
                        "CV RMSE": mean_cv_rmse,
                        "Train R2": mean_train_r2,
                    }
                )

            # =========================================================
            # 4. Baseline leaderboard
            # =========================================================

            results_df = pd.DataFrame(results)

            results_df = results_df.sort_values(
                by="CV R2", ascending=False
            ).reset_index(drop=True)

            print("\n")
            print("=" * 95)
            print("BASELINE MODEL LEADERBOARD")
            print("=" * 95)

            print(
                f"{'Model':<22}"
                f"{'CV R2':>10}"
                f"{'Std':>10}"
                f"{'CV MAE':>16}"
                f"{'CV RMSE':>16}"
                f"{'Train R2':>12}"
            )

            print("-" * 95)

            for _, row in results_df.iterrows():

                print(
                    f"{row['Model']:<22}"
                    f"{row['CV R2']:>10.4f}"
                    f"{row['CV R2 Std']:>10.4f}"
                    f"{row['CV MAE']:>16,.2f}"
                    f"{row['CV RMSE']:>16,.2f}"
                    f"{row['Train R2']:>12.4f}"
                )

            print("=" * 95)

            # =========================================================
            # 5. Tune CatBoost
            # =========================================================

            print("\n")
            print("=" * 95)
            print("CATBOOST HYPERPARAMETER TUNING")
            print("=" * 95)

            logging.info("Starting CatBoost hyperparameter tuning")

            catboost_model = CatBoostRegressor(
                loss_function="RMSE",
                random_seed=42,
                verbose=False,
                allow_writing_files=False,
                thread_count=1,
            )

            catboost_param_grid = {
                "iterations": [200, 300, 500, 700, 1000],
                "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
                "depth": [3, 4, 5, 6],
                "l2_leaf_reg": [1, 3, 5, 7, 10, 20],
                "random_strength": [0, 0.5, 1, 2, 5],
                "bagging_temperature": [0, 0.5, 1, 2],
                "border_count": [32, 64, 128],
            }

            catboost_search = RandomizedSearchCV(
                estimator=catboost_model,
                param_distributions=catboost_param_grid,
                n_iter=40,
                scoring="r2",
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
            )

            catboost_search.fit(X_train, y_train)

            tuned_catboost = catboost_search.best_estimator_

            tuned_catboost_cv_r2 = catboost_search.best_score_

            print("\nBest CatBoost Parameters:")

            for parameter, value in catboost_search.best_params_.items():
                print(f"{parameter}: {value}")

            print(f"\nBest CatBoost CV R2: " f"{tuned_catboost_cv_r2:.4f}")

            logging.info(f"Best CatBoost CV R2: " f"{tuned_catboost_cv_r2:.4f}")

            # =========================================================
            # 6. Compare baseline CatBoost vs tuned CatBoost
            # =========================================================

            baseline_catboost_cv_r2 = results_df.loc[
                results_df["Model"] == "CatBoost", "CV R2"
            ].iloc[0]

            print("\n")
            print("-" * 70)
            print("CATBOOST COMPARISON")
            print("-" * 70)

            print(f"Baseline CatBoost CV R2 : " f"{baseline_catboost_cv_r2:.4f}")

            print(f"Tuned CatBoost CV R2    : " f"{tuned_catboost_cv_r2:.4f}")

            improvement = tuned_catboost_cv_r2 - baseline_catboost_cv_r2

            print(f"Improvement             : " f"{improvement:+.4f}")

            print("-" * 70)

            # =========================================================
            # 7. Select final model
            # =========================================================

            baseline_best_model_name = results_df.iloc[0]["Model"]

            baseline_best_cv_r2 = results_df.iloc[0]["CV R2"]

            if tuned_catboost_cv_r2 > baseline_best_cv_r2:

                final_model = tuned_catboost
                final_model_name = "Tuned CatBoost"
                final_cv_r2 = tuned_catboost_cv_r2

            else:

                final_model = models[baseline_best_model_name]

                final_model_name = baseline_best_model_name

                final_cv_r2 = baseline_best_cv_r2

            # =========================================================
            # 8. Train final selected model
            # =========================================================

            print("\n")
            print("=" * 70)
            print("FINAL MODEL SELECTION")
            print("=" * 70)

            print(f"Selected Model : " f"{final_model_name}")

            print(f"CV R2          : " f"{final_cv_r2:.4f}")

            print("=" * 70)

            logging.info(f"Final selected model: " f"{final_model_name}")

            final_model.fit(X_train, y_train)

            # =========================================================
            # 9. Training performance
            # =========================================================

            y_train_pred = final_model.predict(X_train)

            train_r2 = r2_score(y_train, y_train_pred)

            train_mae = mean_absolute_error(y_train, y_train_pred)

            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

            # =========================================================
            # 10. Holdout validation performance
            # =========================================================

            y_test_pred = final_model.predict(X_test)

            test_r2 = r2_score(y_test, y_test_pred)

            test_mae = mean_absolute_error(y_test, y_test_pred)

            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            # =========================================================
            # 11. Final performance report
            # =========================================================

            print("\n")
            print("=" * 70)
            print("FINAL MODEL PERFORMANCE REPORT")
            print("=" * 70)

            print(f"Selected Model    : " f"{final_model_name}")

            print("-" * 70)

            print(f"CV R2             : " f"{final_cv_r2:.4f}")

            print(f"Train R2          : " f"{train_r2:.4f}")

            print(f"Validation R2     : " f"{test_r2:.4f}")

            print("-" * 70)

            print(f"Train MAE         : " f"{train_mae:,.2f}")

            print(f"Validation MAE    : " f"{test_mae:,.2f}")

            print("-" * 70)

            print(f"Train RMSE        : " f"{train_rmse:,.2f}")

            print(f"Validation RMSE   : " f"{test_rmse:,.2f}")

            print("=" * 70)

            # =========================================================
            # 12. Save leaderboard
            # =========================================================

            os.makedirs("artifacts", exist_ok=True)

            results_df["Tuned CatBoost CV R2"] = np.nan

            results_df.loc[
                results_df["Model"] == "CatBoost", "Tuned CatBoost CV R2"
            ] = tuned_catboost_cv_r2

            results_df.to_csv(
                self.model_trainer_config.leaderboard_file_path, index=False
            )

            logging.info("Model leaderboard saved")

            # =========================================================
            # 13. Save final model
            # =========================================================

            save_object(
                file_path=(self.model_trainer_config.trained_model_file_path),
                obj=final_model,
            )

            logging.info("Final model saved successfully")

            print(
                f"\nBest model saved: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            print(
                f"Leaderboard saved: "
                f"{self.model_trainer_config.leaderboard_file_path}"
            )

            return test_r2

        except Exception as e:

            logging.exception("Error occurred during model training")

            raise CustomException(e, sys)
