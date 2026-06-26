#main aim is to transform the data into a format that can be used for training and testing the model

#imports
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.components.data_ingestion import DataIngestion

from src.exception import custom_exception
from src.logger import logging
from src.utils import save_objects

#we have used this to get content to transform the data
@dataclass
class DataTransformationConfig:
    preprocessing_obj_path = os.path.join('artifacts','preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.Data_transformation_config = DataTransformationConfig()

    def DataTransformationPipeline(self):
        try:
            Ingestion = DataIngestion()

            train_path,test_path = Ingestion.start_data_ingestion()
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            #split the categorical data and numerical data
            nums_category = train_df.select_dtypes(exclude='object').columns
            cat_category = train_df.select_dtypes(include='object').columns

            nums_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat__pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(sparse_output=False)),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('categorical encoding has been done!!!')

            preprocessor = ColumnTransformer(
                [
                    ('nums_pipeline',nums_pipeline,nums_category),
                    ('cat_pipeline',cat__pipeline,cat_category)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise custom_exception(e,sys)
        
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('read train and test data completed')

            target_column_name = "math_score"
            
            input_feature_train = train_df.drop(columns=[target_column_name])
            target_feature_train = train_df[target_column_name]

            input_feature_test = test_df.drop(columns=[target_column_name])
            target_feature_test = test_df[target_column_name]

            nums_category = input_feature_train.select_dtypes(exclude='object').columns
            cat_category = input_feature_train.select_dtypes(include='object').columns

            nums_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            cat__pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('onehotencoder',OneHotEncoder(sparse_output=False)),
                    ('scaler',StandardScaler())
                ]
            )
            logging.info('categorical encoding has been done!!!')

            preprocessor = ColumnTransformer(
                [
                    ('nums_pipeline',nums_pipeline,nums_category),
                    ('cat_pipeline',cat__pipeline,cat_category)
                ]
            )

            logging.info('Applying preprocessing object on training and testing data')

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train)
            input_feature_test_arr = preprocessor.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test)]

            logging.info('saved processing objects')

            save_objects (
                file_path = self.Data_transformation_config.preprocessing_obj_path,
                obj = preprocessor
            )

            return (
                train_arr,
                test_arr,
                self.Data_transformation_config.preprocessing_obj_path
            )           
        
        except Exception as e:
            raise custom_exception(e,sys)