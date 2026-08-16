import sys

import numpy as np
import pandas as pd

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.utils import load_object
from src.house_price_prediction.feature_engineering import add_features


class CustomData:

    def __init__(
        self,
        area: int,
        bedrooms: int,
        bathrooms: int,
        stories: int,
        parking: int,
        mainroad: str,
        guestroom: str,
        basement: str,
        hotwaterheating: str,
        airconditioning: str,
        prefarea: str,
        furnishingstatus: str,
    ):

        self.area = area
        self.bedrooms = bedrooms
        self.bathrooms = bathrooms
        self.stories = stories
        self.parking = parking

        self.mainroad = mainroad
        self.guestroom = guestroom
        self.basement = basement
        self.hotwaterheating = hotwaterheating
        self.airconditioning = airconditioning
        self.prefarea = prefarea
        self.furnishingstatus = furnishingstatus

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                "area": [self.area],
                "bedrooms": [self.bedrooms],
                "bathrooms": [self.bathrooms],
                "stories": [self.stories],
                "parking": [self.parking],
                "mainroad": [self.mainroad],
                "guestroom": [self.guestroom],
                "basement": [self.basement],
                "hotwaterheating": [self.hotwaterheating],
                "airconditioning": [self.airconditioning],
                "prefarea": [self.prefarea],
                "furnishingstatus": [self.furnishingstatus],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:

            raise CustomException(e, sys)


class PredictPipeline:

    def __init__(self):

        self.model_path = "artifacts/model.pkl"

        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame):

        try:

            # ==========================================
            # Load trained objects
            # ==========================================

            model = load_object(file_path=self.model_path)

            preprocessor = load_object(file_path=self.preprocessor_path)

            # ==========================================
            # Apply feature engineering
            # ==========================================

            features = add_features(features)

            # ==========================================
            # Apply preprocessing
            # ==========================================

            transformed_data = preprocessor.transform(features)

            # ==========================================
            # Predict log(price)
            # ==========================================

            predicted_log_price = model.predict(transformed_data)

            # ==========================================
            # Convert back to original price
            # ==========================================

            predicted_price = np.expm1(predicted_log_price)

            return predicted_price

        except Exception as e:

            raise CustomException(e, sys)
