import os
import sys
from src.exception import CustomException
from src.log import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            bmw=pd.read_csv("note/car_data/bmw.csv")
            audi=pd.read_csv("note/car_data/audi.csv")
            ford=pd.read_csv("note/car_data/ford.csv")
            hyundi=pd.read_csv("note/car_data/hyundi.csv")
            benz=pd.read_csv("note/car_data/merc.csv")
            toyota=pd.read_csv("note/car_data/toyota.csv")

            hyundi.rename(columns={'tax(£)': 'tax'}, inplace=True)

            frames = [bmw,audi,ford,hyundi,benz,toyota]
            merged = pd.concat(frames)
            df = merged.drop_duplicates()
            logging.info("Read the datasets as dataframes")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train Test Split initiated")
            training_data,testing_data=train_test_split(df,test_size=0.2)

            training_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            testing_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion Completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()