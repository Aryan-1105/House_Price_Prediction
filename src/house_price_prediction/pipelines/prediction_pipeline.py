import sys

import pandas as pd

from src.house_price_prediction.exception import CustomException
from src.house_price_prediction.utils import load_object

class CustomData:
    """
    Takes individual house feature values (e.g. from a web form)
    and converts them into a pandas DataFrame with the same column
    names used during training, so the preprocessor can handle it.
    """

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
        """
        Builds a single-row DataFrame from the stored values.
        Column names and order don't have to match the preprocessor's
        internal order exactly (ColumnTransformer selects by name),
        but the NAMES must match exactly.
        """
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
    """
    Loads the trained model and preprocessor from disk,
    then uses them to predict the price for new house data.
    """

    def __init__(self):
        self.model_path = "artifacts/model.pkl"
        self.preprocessor_path = "artifacts/preprocessor.pkl"

    def predict(self, features: pd.DataFrame):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)

            # Apply the SAME transformation used during training
            data_scaled = preprocessor.transform(features)

            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)