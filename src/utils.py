import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl

from src.logger import logging
from src.exception import custom_exception
from sklearn.metrics import r2_score

def save_objects(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pkl.dump(obj,file_obj)

        logging.info(f'pickle file has been saved at {dir}')
    
    except Exception as e:
        raise custom_exception(e,sys)
    

def evaluation_model(x_train,x_test,y_train,y_test,models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]

            model.fit(x_train,y_train)

            y_train_predict = model.predict(x_train)
            y_test_predict = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_predict)
            test_model_score = r2_score(y_test,y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise custom_exception(e,sys)