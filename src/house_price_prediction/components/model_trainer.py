import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging
from src.house_price_prediction.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains a Gradient Boosting model on the transformed train array,
        evaluates it on the test array, and saves the trained model.

        train_array / test_array: numpy arrays where the LAST column is
        the target (price) and all other columns are features. These come
        directly from DataTransformation.initiate_data_transformation().
        """
        try:
            logging.info("Splitting training and test input data")

            # Last column is the target (price); everything else is features
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # This was the winning model from Model_Trainer.ipynb comparison
            model = GradientBoostingRegressor(random_state=42)

            logging.info("Training Gradient Boosting model")
            model.fit(X_train, y_train)

            y_test_pred = model.predict(X_test)
            test_r2 = r2_score(y_test, y_test_pred)

            logging.info(f"Test R2 score: {test_r2}")

            # Save the trained model so prediction_pipeline.py can load it later
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path, obj=model
            )

            return test_r2

        except Exception as e:
            raise CustomException(e, sys)
