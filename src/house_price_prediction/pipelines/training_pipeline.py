import sys

from src.house_price_prediction.components.data_ingestion import DataIngestion
from src.house_price_prediction.components.data_transformation import DataTransformation
from src.house_price_prediction.components.model_trainer import ModelTrainer
from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.logger import logging


def start_training_pipeline():
    """
    Runs the full training pipeline:
    1. Ingest data from PostgreSQL, split into train/test CSVs
    2. Transform train/test data using the preprocessor
    3. Train the model and save it

    Returns the final test R2 score.
    """
    try:
        logging.info("Training pipeline started")

        # Step 1: Data Ingestion
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = (
            data_transformation.initiate_data_transformation(train_path, test_path)
        )

        # Step 3: Model Training
        model_trainer = ModelTrainer()
        test_r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Training pipeline completed. Test R2 score: {test_r2}")

        return test_r2

    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    r2_score = start_training_pipeline()
    print("Training pipeline finished successfully.")
    print("Final Test R2 Score:", r2_score)
