import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP:str = datetime.now().strftime("%m_%d_%y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = PIPELINE_NAME
    artifact_dir:str = os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str = TIMESTAMP
    
training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    data_ingestion_ingested_dir: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR)
    feature_store_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name: str = DATA_INGESTION_COLLECTION_NAME
    
@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)

@dataclass
class DataTransformationConfig:
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    data_transformation_transformed_dir:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
    input_feature_final_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,DATA_TRANSFORMATION_INPUT_FEATURE_FINAL_PATH)
    target_feature_final_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,DATA_TRANSFORMATION_TARGET_FEATURE_FINAL_PATH)
    transformed_object_file_path:str = os.path.join(data_transformation_dir,
                                                    DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                    PREPROCESSING_OBJECT_FILE_NAME)
    
@dataclass
class ModelTrainerConfig:
    model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
    model_trainer_trained_model_file_path:str = os.path.join(model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_TRAINER_TRAINED_MODEL_NAME)
    expected_accuracy:float = MODEL_TRAINER_EXPECTED_SCORE
    model_trainer_config_file_path:str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH
    train_test_split_ratio:str = MODEL_TRAINER_TRAIN_TEST_SPLIT_RATIO
    model_trainer_random_state:int = MODEL_TRAINER_RANDOM_STATE
    _weights:str = MODEL_TRAINER_WEIGHTS
    _n_neighbors:int = MODEL_TRAINER_N_NEIGHBORS
    _algorithm:str = MODEL_TRAINER_ALGORITHM