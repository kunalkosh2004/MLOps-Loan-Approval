import sys
from typing import Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import load_numpy_array_data, save_object, load_object
from src.entity.estimator import MyModel
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact, ClassificationMetricArtifact


class ModelTrainer:
    def __init__(self, data_transformation_artifact:DataTransformationArtifact,
                 model_trainer_config:ModelTrainerConfig):
        """
        :param data_transformation_artifact: Output reference of data transformation artifact stage
        :param model_trainer_config: Configuration for model training
        """
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise MyException(e,sys)
        
    def get_model_object_and_report(self, train_arr: np.array, test_arr: np.array) -> Tuple[object,object]:
        """
        Method Name :   get_model_object_and_report
        Description :   This function trains a KNeighborsClassifier with specified parameters
        
        Output      :   Returns metric artifact object and trained model object
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Training KNeighborsClassifier with specified params")
            
            # splitting the train and test data into feature and target column
            x_train, y_train, x_test, y_test = train_arr[:,:-1], train_arr[:,-1], test_arr[:,:-1], test_arr[:,-1]
            logging.info("train-test split done")
            
            # Initializing KNeighborsClassifier with specied params
            params = {'KNN': {'weights': self.model_trainer_config._weights,
                              'n_neighbors': self.model_trainer_config._n_neighbors,
                              'algorithm': self.model_trainer_config._algorithm}}
            clf = KNeighborsClassifier(**params['KNN'])
            
            # Training the model
            logging.info("Training the model...")
            clf.fit(x_train,y_train)
            logging.info("Model Trained")
            
            # Doing Predictions and evaluations
            y_pred = clf.predict(x_test)
            accuracy= accuracy_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            print(f"Accuracy Score:- {accuracy}")
            
            # Creating Metric artifact
            metric_artifact = ClassificationMetricArtifact(accuracy_score=accuracy,f1_score=f1,
                                                           precision_score=precision,recall_score=recall)
            return clf,metric_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer of ModelTrainer class")
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates the model training steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print("------------------------------------------------------------------------------------------------")
            print("Starting Model Trainer Component")
            # Load Transformed train and test data
            train_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path = self.data_transformation_artifact.transformed_test_file_path)
            logging.info("train-test data loaded")
            
            # Train model and get metrics
            trained_model, model_artifact = self.get_model_object_and_report(train_arr = train_arr, test_arr = test_arr)
            logging.info("Model object and artifact loaded")
            
            # load_preprocessing object
            preprocessing_obj = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)
            logging.info("Preprocessing object loaded")
            
            # Check if model's acc meets the expected threshold
            if accuracy_score(train_arr[:,-1], trained_model.predict(train_arr[:,:-1]))<self.model_trainer_config.expected_accuracy:
                logging.info("No model found with score aborve the base score")
                raise Exception("No model found with score aborve the base score")
            
            # Save the final model object that includes both preprocessign and the trained model
            logging.info("Saving new model as performance is better than previous one. ")
            my_model = MyModel(preprocessing_object=preprocessing_obj, trained_model_object=trained_model)
            save_object(file_path = self.model_trainer_config.model_trainer_trained_model_file_path, obj = my_model)
            
            logging.info("Saved final model object that includes both preprocessing and the trained model")
            
            # Check and return the ModelTrainerArtifact
            model_trainer_artifact = ModelTrainerArtifact(transformed_model_file_path=self.model_trainer_config.model_trainer_trained_model_file_path,
                                                          model_artifact = model_artifact)
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        
        except Exception as e:
            raise MyException(e,sys)

            
            
