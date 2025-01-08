import sys
import os
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH,CURRENT_YEAR
from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file

class DataTransformation:
    def __init__(self, data_ingestion_artifact = DataIngestionArtifact,
                       data_validation_artifact = DataValidationArtifact,
                       data_transformation_config = DataTransformationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(file_path = SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    @staticmethod
    def read_data(file_path:str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformer object for the data, 
        including gender mapping, dummy variable creation, column renaming,
        feature scaling, and type adjustments.
        """
        logging.info("Entered the get_data_transformer_object of DataTransformation class")
        try:
            # Initialize Transformers
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            or_transformer = OrdinalEncoder()
            pt = PowerTransformer(method = "yeo-johnson")
            
            transform_pipe = Pipeline(steps=[
                                ('transformer', PowerTransformer(method='yeo-johnson'))
                            ])
            or_columns = self._schema_config['or_columns']
            oh_columns = self._schema_config['oh_columns']
            transform_features = self._schema_config['transform_features']
            num_features = self._schema_config['numerical_columns']
            
            preprocessor = ColumnTransformer([
                            ('OneHotEncoder',oh_transformer,oh_columns),
                            ('OrdinalEncoder',or_transformer,or_columns),
                            ('Transformer',transform_pipe,transform_features),
                            ('StandardScaler',numeric_transformer,num_features)
                        ],remainder="passthrough")
            
            final_pipeline = Pipeline(steps=[("preprocessor",preprocessor)])
            logging.info("Final Pipeline Ready!")
            logging.info("Exited get_data_transformer_object of DataTransformer class")
            return final_pipeline
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)
            
            train_df = self.read_data(file_path = self.data_ingestion_artifact.training_file_path)
            test_df = self.read_data(file_path = self.data_ingestion_artifact.testing_file_path)
            logging.info("Train and test data are loaded")
            
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input ans Target columns defines for both train and test df.")
            
            logging.info("Starting Data Transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")
            
            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            logging.info("Applying SMOOTEENN for handling imbalanced dataset")
            smt = SMOTEENN(sampling_strategy = 'minority')
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr,target_feature_train_df
            )
            
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr,target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")
            
            train_arr = np.c_[input_feature_train_final,np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_final,np.array(target_feature_test_final)]
            logging.info("Feature-target concatenation done for train-test df.")
            
            save_object(obj=preprocessor, file_path = self.data_transformation_config.transformed_object_file_path)
            
            dir_name = os.path.join(self.data_transformation_config.data_transformation_transformed_dir)
            os.makedirs(dir_name, exist_ok=True)
            
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path,array=test_arr)
            logging.info("Saving transformation object and transformed files.")

            logging.info("Data transformation completed successfully")
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path = self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path = self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path = self.data_transformation_config.transformed_object_file_path
            )
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e,sys)
            
