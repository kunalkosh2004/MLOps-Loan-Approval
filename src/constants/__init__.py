import os
from datetime import date

# For MongoDB Connection
DATABASE_NAME = 'Loan'
COLLECTION_NAME = 'Loan-Data'
MONGODB_URL_KEY = 'MONGODB_URL'

PIPELINE_NAME:str = ""
ARTIFACT_DIR:str = "artifact"

MODEL_FILE_NAME = 'model.pkl'
FILE_NAME:str = 'laon_data.csv'

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