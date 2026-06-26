import os
import sys
import pandas as pd
import numpy as np
import pickle as pkl

from src.logger import logging
from src.exception import custom_exception


def save_objects(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,'wb') as file_obj:
            pkl.dump(obj,file_obj)

        logging.info(f'pickle file has been saved at {dir}')
    
    except Exception as e:
        raise custom_exception(e,sys)