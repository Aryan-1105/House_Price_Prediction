import os
import sys
import dill

import pandas as pd
from sqlalchemy import create_engine

from dotenv import load_dotenv

from src.house_price_prediction.logger import logging
from src.house_price_prediction.exception import CustomException

# ==========================================================
# Load environment variables
# ==========================================================

load_dotenv()


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_TABLE = os.getenv("DB_TABLE")


# ==========================================================
# Read data from PostgreSQL
# ==========================================================


def read_sql_data():
    """
    Reads the complete dataset from PostgreSQL
    and returns it as a pandas DataFrame.
    """

    logging.info("Reading SQL database started")

    try:

        # --------------------------------------------------
        # Check required environment variables
        # --------------------------------------------------

        required_variables = {
            "DB_HOST": DB_HOST,
            "DB_PORT": DB_PORT,
            "DB_NAME": DB_NAME,
            "DB_USER": DB_USER,
            "DB_PASSWORD": DB_PASSWORD,
            "DB_TABLE": DB_TABLE,
        }

        missing_variables = [
            key for key, value in required_variables.items() if not value
        ]

        if missing_variables:
            raise ValueError(f"Missing environment variables: {missing_variables}")

        # --------------------------------------------------
        # Create PostgreSQL connection
        # --------------------------------------------------

        engine = create_engine(
            f"postgresql+psycopg2://"
            f"{DB_USER}:{DB_PASSWORD}@"
            f"{DB_HOST}:{DB_PORT}/"
            f"{DB_NAME}"
        )

        logging.info("PostgreSQL connection established")

        # --------------------------------------------------
        # Read data
        # --------------------------------------------------

        query = f'SELECT * FROM "{DB_TABLE}"'

        df = pd.read_sql(query, engine)

        logging.info(f"Data loaded successfully. Shape: {df.shape}")

        # --------------------------------------------------
        # Close database connection
        # --------------------------------------------------

        engine.dispose()

        return df

    except Exception as ex:

        logging.exception("Error while reading SQL database")

        raise CustomException(ex, sys)


# ==========================================================
# Save Python object
# ==========================================================


def save_object(file_path, obj):
    """
    Saves a Python object such as:
    - trained ML model
    - preprocessor
    - pipeline

    using dill serialization.
    """

    try:

        # Get directory from file path
        dir_path = os.path.dirname(file_path)

        # Create directory if necessary
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save object
        with open(file_path, "wb") as file_obj:

            dill.dump(obj, file_obj)

        logging.info(f"Object saved successfully: {file_path}")

    except Exception as e:

        logging.exception(f"Error saving object: {file_path}")

        raise CustomException(e, sys)


# ==========================================================
# Load Python object
# ==========================================================


def load_object(file_path):
    """
    Loads a Python object that was previously
    saved using save_object().
    """

    try:

        with open(file_path, "rb") as file_obj:

            obj = dill.load(file_obj)

        logging.info(f"Object loaded successfully: {file_path}")

        return obj

    except Exception as e:

        logging.exception(f"Error loading object: {file_path}")

        raise CustomException(e, sys)
