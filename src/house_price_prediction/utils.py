import os
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
        engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        logging.info("Connection Established",engine)
        df = pd.read_sql(f"SELECT * FROM {DB_TABLE}",engine)
        print(df.head())
        
        
        return df 
    
      
    except Exception as ex:
        raise CustomException(ex, sys)