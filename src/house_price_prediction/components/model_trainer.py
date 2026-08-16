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

    leaderboard_file_path: str = os.path.join("artifacts", "model_comparison_cv.csv")


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
            # 2. Log-transform target
            # =========================================================

            y_train_log = np.log1p(y_train)

            y_test_log = np.log1p(y_test)

            logging.info("Target log transformation completed")

            # =========================================================
            # 3. Define baseline models
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
            # 4. 5-Fold Cross Validation
            # =========================================================

            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "mse": "neg_mean_squared_error",
            }

            results = []

            print("\n")
            print("=" * 100)
            print("MODEL COMPARISON - 5 FOLD CROSS VALIDATION")
            print("=" * 100)

            for model_name, model in models.items():

                print(f"\nEvaluating: {model_name}")

                logging.info(f"Evaluating {model_name}")

                cv_results = cross_validate(
                    estimator=model,
                    X=X_train,
                    y=y_train_log,
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
                        "CV R2 (Log)": mean_cv_r2,
                        "CV R2 Std": std_cv_r2,
                        "CV MAE (Log)": mean_cv_mae,
                        "CV RMSE (Log)": mean_cv_rmse,
                        "Train R2 (Log)": mean_train_r2,
                    }
                )

            # =========================================================
            # 5. Create CV leaderboard
            # =========================================================

            results_df = pd.DataFrame(results)

            results_df = results_df.sort_values(
                by="CV R2 (Log)", ascending=False
            ).reset_index(drop=True)

            print("\n")
            print("=" * 100)
            print("BASELINE MODEL LEADERBOARD")
            print("=" * 100)

            print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

            print("=" * 100)

            # =========================================================
            # 6. Identify best baseline model
            # =========================================================

            best_baseline_row = results_df.iloc[0]

            best_baseline_model_name = best_baseline_row["Model"]

            best_baseline_cv_r2 = best_baseline_row["CV R2 (Log)"]

            print("\n")
            print("=" * 70)
            print("BEST BASELINE MODEL")
            print("=" * 70)

            print(f"Model : {best_baseline_model_name}")

            print(f"CV R2 : {best_baseline_cv_r2:.4f}")

            print("=" * 70)

            # =========================================================
            # 7. Lasso Hyperparameter Tuning
            # =========================================================

            print("\n")
            print("=" * 100)
            print("LASSO HYPERPARAMETER TUNING")
            print("=" * 100)

            logging.info("Starting Lasso hyperparameter tuning")

            lasso_model = Lasso(max_iter=10000)

            lasso_param_grid = {"alpha": np.logspace(-5, 1, 50)}

            lasso_search = RandomizedSearchCV(
                estimator=lasso_model,
                param_distributions=lasso_param_grid,
                n_iter=30,
                scoring="r2",
                cv=cv,
                random_state=42,
                n_jobs=-1,
                verbose=1,
                return_train_score=True,
            )

            lasso_search.fit(X_train, y_train_log)

            tuned_lasso = lasso_search.best_estimator_

            tuned_lasso_cv_r2 = lasso_search.best_score_

            print("\nBest Lasso Parameters:")

            for parameter, value in lasso_search.best_params_.items():
                print(f"{parameter}: {value}")

            print(f"\nBest Lasso CV R2: " f"{tuned_lasso_cv_r2:.4f}")

            logging.info(f"Best Lasso CV R2: " f"{tuned_lasso_cv_r2:.4f}")

            # =========================================================
            # 8. Baseline vs Tuned Lasso
            # =========================================================

            baseline_lasso_cv_r2 = results_df.loc[
                results_df["Model"] == "Lasso Regression", "CV R2 (Log)"
            ].iloc[0]

            improvement = tuned_lasso_cv_r2 - baseline_lasso_cv_r2

            print("\n")
            print("-" * 70)
            print("LASSO TUNING RESULT")
            print("-" * 70)

            print(f"Baseline Lasso CV R2 : " f"{baseline_lasso_cv_r2:.4f}")

            print(f"Tuned Lasso CV R2    : " f"{tuned_lasso_cv_r2:.4f}")

            print(f"Improvement          : " f"{improvement:+.4f}")

            print("-" * 70)

            # =========================================================
            # 9. Final model selection
            # =========================================================

            if (
                best_baseline_model_name == "Lasso Regression"
                and tuned_lasso_cv_r2 > baseline_lasso_cv_r2
            ):

                final_model = tuned_lasso

                final_model_name = "Tuned Lasso Regression"

                final_cv_r2 = tuned_lasso_cv_r2

            else:

                final_model = models[best_baseline_model_name]

                final_model_name = best_baseline_model_name

                final_cv_r2 = best_baseline_cv_r2

            print("\n")
            print("=" * 70)
            print("FINAL MODEL SELECTION")
            print("=" * 70)

            print(f"Selected Model : " f"{final_model_name}")

            print(f"CV R2          : " f"{final_cv_r2:.4f}")

            print("=" * 70)

            logging.info(f"Final selected model: " f"{final_model_name}")

            # =========================================================
            # 10. Train final model
            # =========================================================

            final_model.fit(X_train, y_train_log)

            logging.info("Final model fitted successfully")

            # =========================================================
            # 11. Predictions in log space
            # =========================================================

            predicted_log_price_train = final_model.predict(X_train)

            predicted_log_price_test = final_model.predict(X_test)

            # =========================================================
            # 12. Convert predictions back to price
            # =========================================================

            predicted_price_train = np.expm1(predicted_log_price_train)

            predicted_price_test = np.expm1(predicted_log_price_test)

            # =========================================================
            # 13. Final metrics
            # =========================================================

            train_r2 = r2_score(y_train, predicted_price_train)

            test_r2 = r2_score(y_test, predicted_price_test)

            train_mae = mean_absolute_error(y_train, predicted_price_train)

            test_mae = mean_absolute_error(y_test, predicted_price_test)

            train_rmse = np.sqrt(mean_squared_error(y_train, predicted_price_train))

            test_rmse = np.sqrt(mean_squared_error(y_test, predicted_price_test))

            # =========================================================
            # 14. Final performance report
            # =========================================================

            print("\n")
            print("=" * 75)
            print("FINAL MODEL PERFORMANCE")
            print("=" * 75)

            print(f"Model        : " f"{final_model_name}")

            print(f"CV R2        : " f"{final_cv_r2:.4f}")

            print(f"Train R2     : " f"{train_r2:.4f}")

            print(f"Test R2      : " f"{test_r2:.4f}")

            print(f"Train MAE    : " f"₹{train_mae:,.2f}")

            print(f"Test MAE     : " f"₹{test_mae:,.2f}")

            print(f"Train RMSE   : " f"₹{train_rmse:,.2f}")

            print(f"Test RMSE    : " f"₹{test_rmse:,.2f}")

            print("=" * 75)

            # =========================================================
            # 15. Save leaderboard
            # =========================================================

            os.makedirs("artifacts", exist_ok=True)

            results_df.to_csv(
                self.model_trainer_config.leaderboard_file_path, index=False
            )

            logging.info("CV leaderboard saved successfully")

            # =========================================================
            # 16. Save model
            # =========================================================

            save_object(
                file_path=(self.model_trainer_config.trained_model_file_path),
                obj=final_model,
            )

            logging.info("Final model saved successfully")

            print(
                "\nFinal model saved: "
                f"{self.model_trainer_config.trained_model_file_path}"
            )

            print(
                "Leaderboard saved: "
                f"{self.model_trainer_config.leaderboard_file_path}"
            )

            return test_r2

        except Exception as e:

            logging.exception("Error occurred during model training")

            raise CustomException(e, sys)
