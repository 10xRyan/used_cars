import os
import sys
import dill

import numpy as np
import pandas as pd

from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            #para=param[list(models.keys())[i]]

            #gs = GridSearchCV(model,para,cv=3)
            #gs.fit(X_train,y_train)

            #model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_test_pred = model.predict(X_test)

            #!Criteria for selecting the best model: mea aboslute error (could be changed)
            #Needs to change the method to min() in model trainer
            test_model_score = mean_absolute_error(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)