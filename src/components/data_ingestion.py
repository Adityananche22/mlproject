import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifact', 'train.csv')
    test_data_path: str=os.path.join('artifact', 'test.csv')
    raw_data_path: str=os.path.join('artifact', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        print("Inside initiate_data_ingestion")

        try:
            df = pd.read_csv('notebook/data/stud.csv')

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=True, header=True)

            train_set, test_set = train_test_split(
                df, test_size=0.2, random_state=42
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=True, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=True, header=True)

        except Exception as e:
            raise customException(e, sys)
            
if __name__=="__main__":
    print("Program started")

    obj = DataIngestion()

    print("Calling ingestion method")
    obj.initiate_data_ingestion()

    print("Program completed")

    