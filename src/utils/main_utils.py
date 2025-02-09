import os
import sys
import pandas as pd
import numpy as np
import dill
import yaml

from src.logger import logging
from src.exception import MyException

def read_yaml_file(file_path:str) -> dict:
    try:
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(e,sys)
    
def write_yaml_file(file_path:str, content:object, replace:bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exits_ok=True)
        with open(file_path, 'wb') as file:
            yaml.write(file)
    except Exception as e:
        raise MyException(e,sys)
    
def load_object(file_path:str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, 'rb') as file:
            obj = dill.load(file)
        return obj
    except Exception as e:
        raise MyException(e,sys)
    
def save_numpy_array_data(file_path:str, array: np.array) -> None:
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        with open(file_path,'wb') as file:
            np.save(file,array)
    except Exception as e:
        raise MyException(e,sys)
    
def load_numpy_array_data(file_path:str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path,'rb') as file:
            return np.load(file)
    except Exception as e:
        raise MyException(e,sys)
    
def save_object(file_path:str, obj:object) -> None:
    logging.info("Entered the save_object method of utils")
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj,file) 
        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise MyException(e,sys)