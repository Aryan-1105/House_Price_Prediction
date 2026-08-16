import os
import dill
import sys

import pandas as pd
from sqlalchemy import create_engine

from src.house_price_prediction.logger import logging
from src.house_price_prediction.exception import CustomException


from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TABLE = os.getenv("DB_TABLE")


def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        engine = create_engine(
            f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        logging.info(f"Connection Established: {engine}")
        df = pd.read_sql(f"SELECT * FROM {DB_TABLE}", engine)
        print(df.head())

        return df

    except Exception as ex:
        raise CustomException(ex, sys)


def save_object(file_path, obj):
    """
    Saves any Python object (like a trained model or a preprocessor)
    to a file, so we can reuse it later without rebuilding it.
    """
    try:
        # Get the folder path from the full file path
        dir_path = os.path.dirname(file_path)

        # Create that folder if it doesn't already exist
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in "write binary" mode and save the object into it
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    """
    Loads a Python object (like a trained model or a preprocessor)
    back from a file that was created using save_object().
    """
    try:
        # Open the file in "read binary" mode and load the object from it
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
