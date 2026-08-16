import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from src.house_price_prediction.feature_engineering import add_features
from src.house_price_prediction.logger import logging
from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.utils import save_object


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = (
            DataTransformationConfig()
        )

    def get_data_transformer_object(self):

        try:

            numerical_columns = [
                "area",
                "bedrooms",
                "bathrooms",
                "stories",
                "parking",
                "total_rooms",
                "area_per_bedroom",
                "area_per_bathroom",
                "amenity_count"
            ]

            categorical_columns = [
                "mainroad",
                "guestroom",
                "basement",
                "hotwaterheating",
                "airconditioning",
                "prefarea",
                "furnishingstatus"
            ]

            num_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )

            logging.info(
                f"Numerical columns: {numerical_columns}"
            )

            logging.info(
                f"Categorical columns: {categorical_columns}"
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns
                    ),
                    (
                        "cat_pipeline",
                        cat_pipeline,
                        categorical_columns
                    )
                ]
            )

            return preprocessor

        except Exception as e:

            raise CustomException(
                e,
                sys
            )

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:

            train_df = pd.read_csv(
                train_path
            )

            test_df = pd.read_csv(
                test_path
            )

            logging.info(
                "Train and test data loaded"
            )

            # ==========================================
            # Feature Engineering
            # ==========================================

            train_df = add_features(
                train_df
            )

            test_df = add_features(
                test_df
            )

            logging.info(
                "Feature engineering completed"
            )

            target_column_name = "price"

            # ==========================================
            # Separate Features and Target
            # ==========================================

            input_feature_train_df = (
                train_df.drop(
                    columns=[target_column_name]
                )
            )

            target_feature_train_df = (
                train_df[target_column_name]
            )

            input_feature_test_df = (
                test_df.drop(
                    columns=[target_column_name]
                )
            )

            target_feature_test_df = (
                test_df[target_column_name]
            )

            # ==========================================
            # Preprocessing
            # ==========================================

            preprocessing_obj = (
                self.get_data_transformer_object()
            )

            logging.info(
                "Fitting preprocessor on training data"
            )

            input_feature_train_arr = (
                preprocessing_obj.fit_transform(
                    input_feature_train_df
                )
            )

            input_feature_test_arr = (
                preprocessing_obj.transform(
                    input_feature_test_df
                )
            )

            # ==========================================
            # Combine Features + Target
            # ==========================================

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(
                    target_feature_train_df
                )
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(
                    target_feature_test_df
                )
            ]

            # ==========================================
            # Save Preprocessor
            # ==========================================

            save_object(
                file_path=(
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                ),
                obj=preprocessing_obj
            )

            logging.info(
                "Preprocessor saved successfully"
            )

            return (
                train_arr,
                test_arr,
                (
                    self
                    .data_transformation_config
                    .preprocessor_obj_file_path
                )
            )

        except Exception as e:

            raise CustomException(
                e,
                sys
            )