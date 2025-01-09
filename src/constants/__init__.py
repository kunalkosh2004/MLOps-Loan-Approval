import os
from datetime import date

# For MongoDB Connection
DATABASE_NAME = 'Loan'
COLLECTION_NAME = 'Loan-Data'
MONGODB_URL_KEY = 'MONGODB_URL'

PIPELINE_NAME:str = ""
ARTIFACT_DIR:str = "artifact"
CURRENT_YEAR:str = date.today().year

MODEL_FILE_NAME = 'model.pkl'
FILE_NAME:str = 'laon_data.csv'
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

TARGET_COLUMN = 'loan_status'
TRAIN_FILE_NAME:str = 'train.csv'
TEST_FILE_NAME:str = 'test.csv'
SCHEMA_FILE_PATH:str = os.path.join('config','schema.yaml')

# Skipped the AWS Constants

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME:str = 'Loan-Data'
DATA_INGESTION_DIR_NAME:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str = 'feature_store'
DATA_INGESTION_INGESTED_DIR:str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO:float = 0.2

"""
Data Validation related constant start with DATA_INGESTION VAR NAME
"""
DATA_VALIDATION_DIR_NAME:str = 'data_validation'
DATA_VALIDATION_REPORT_FILE_NAME:str = 'report.yaml'

"""
Data Transformation related constant start with DATA_INGESTION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME:str = 'data_transformation'
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = 'transformed'
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = 'transformed_object'

"""
Model Trainer related constant start with DATA_INGESTION VAR NAME
"""
MODEL_TRAINER_DIR_NAME:str = 'model_trainer'
MODEL_TRAINER_TRAINED_MODEL_DIR:str = 'trained_model'
MODEL_TRAINER_TRAINED_MODEL_NAME:str = 'model.pkl'
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH:str = os.path.join('config','model.yaml')
MODEL_TRAINER_EXPECTED_SCORE:float = 0.8
MODEL_TRAINER_WEIGHTS:str = 'distance'
MODEL_TRAINER_N_NEIGHBORS:int = 3
MODEL_TRAINER_ALGORITHM:str = 'auto'