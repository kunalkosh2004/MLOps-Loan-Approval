import os
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import MyException
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.loan_data import LoanData

class DataIngestion:
    
    def __init__(self, data_ingestion_config:DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from MongoDB")
            my_data = LoanData()
            dataframe = my_data.export_collection_as_dataframe(collection_name = self.data_ingestion_config.collection_name)
            logging.info(f"Shape of DataFrame:{dataframe.shape}")
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Saving exported data into feature store: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path, header=True, index=False)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)
        
    def split_data_as_train_test(self,dataframe: pd.DataFrame) -> None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test of Data_Ingestion class")
        try:
            train_df, test_df = train_test_split(dataframe,test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info("Exited split_data_as_train_test of Data_Ingestion class")
            
            dir_name = self.data_ingestion_config.data_ingestion_ingested_dir
            os.makedirs(dir_name,exist_ok=True)
            
            logging.info(f"Exporting train and test file path")
            train_df.to_csv(self.data_ingestion_config.training_file_path,header=True,index=False)
            test_df.to_csv(self.data_ingestion_config.testing_file_path,header=True,index=False)
            logging.info(f"Exported train and test file path.")
            
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initaite_data_ingestion of Data_Ingestion class")
        
        try:
            dataframe = self.export_data_into_feature_store()
            logging.info("Got the data from MongoDB")
            
            self.split_data_as_train_test(dataframe)
            
            logging.info("Performed train_test_split om the dataset")
            
            logging.info("Exited from initaite_data_ingestion of Data_Ingestion class")
            data_ingestion_artifact = DataIngestionArtifact(training_file_path = self.data_ingestion_config.training_file_path,
                                                            testing_file_path = self.data_ingestion_config.testing_file_path,
                                                            feature_store_file_path = self.data_ingestion_config.feature_store_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
            
        
        
