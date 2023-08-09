import sys
import pandas as pd
from src.exception import CustomException
from src.utility import load_obj

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            preprocessor_path="artifacts/preprocessor.pkl"
            model=load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            data_processed = preprocessor.transform(features)
            pred=model.predict(data_processed)
            return pred
        
        except Exception as e:
            raise CustomException(e,sys)
    

class CustomData:
    def __init__(self,
                 model:str,
                 year:int,
                 transmission:str,
                 mileage:int,
                 fuelType:str,
                 tax: int,
                 mpg:int,
                 engineSize: int):
        self.model=model
        self.year=year
        self.transmission=transmission
        self.mileage=mileage
        self.fuelType=fuelType
        self.tax=tax
        self.mpg=mpg
        self.engineSize=engineSize

    def get_data_as_df(self):
        try:
            custom_data_dict={
                "model":[self.model],
                "year":[self.year],
                "transmission":[self.transmission],
                "mileage":[self.mileage],
                "fuelType":[self.fuelType],
                "tax":[self.tax],
                "mpg":[self.mpg],
                "engineSize":[self.engineSize]
            }
            print(custom_data_dict)
            return pd.DataFrame(custom_data_dict)
        
        except Exception as e:
            raise CustomException(e,sys)
        