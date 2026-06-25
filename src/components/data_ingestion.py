#aim is to read the data from the source and load it into the feature store

#imports
import sys
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.exception import custom_exception
from src.logger import logging
from dataclasses import dataclass

#creating a data ingestion class
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts','train.csv') #path where my data will be saved
    test_data_path:str = os.path.join('artifacts','test.csv')
    raw_data_path:str = os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    #data loading and ingestion pipeline
    def start_data_ingestion(self): #to read the data from various sources
        logging.info('Entered the data ingestion component')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('data loaded sucessfully')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("data ingestion completed for raw data")
            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('data ingestion completed for train and test data"')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise custom_exception(e,sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.start_data_ingestion()