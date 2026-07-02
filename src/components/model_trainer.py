import os
import sys
from dataclasses import dataclass

#sklearn for regression 
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_objects
from src.utils import evaluation_model

#config class
@dataclass
class model_trainer_config():
    trained_model_file_path = os.path.join('artifacts','model.pkl')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def start_model_training(self,train_arr,test_arr):
        try:
            logging.info('spliting train and test')
            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]

            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            models = {
            "Linear Regression": LinearRegression(),
            "Lasso": Lasso(),
            "Ridge": Ridge(),
            "K-Neighbors Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(), 
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor()
            }

            model_report:dict=evaluation_model(x_train=x_train,x_test=x_test,y_train=y_train,y_test=y_test,models=models)

            best_model_score = (max(sorted(model_report.values())))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise custom_exception('No best model found! all the model accuracy score are below 60%')
            
            logging.info(f"best model found for both training and test data best model - {best_model_name} , accuracy score - {best_model_score}")
            
            save_objects(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            logging.info('best model as been saved at {file_path}')
            print(model_report)
        
        except Exception as e:
            raise custom_exception(e,sys)