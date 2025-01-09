from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    training_file_path:str
    testing_file_path:str
    feature_store_file_path:str

@dataclass
class DataValidationArtifact:
    validation_status:bool
    message:str
    validation_report_file_path:str
    
@dataclass
class DataTransformationArtifact:
    transformed_object_file_path:str
    input_feature_final_path:str
    target_feature_final_path:str
    
@dataclass
class ClassificationMetricArtifact:
    accuracy_score:float
    f1_score:float
    precision_score:float
    recall_score:float
    
@dataclass
class ModelTrainerArtifact:
    transformed_model_file_path:str
    model_artifact:ClassificationMetricArtifact