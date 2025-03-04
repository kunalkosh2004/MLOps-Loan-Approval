import sys
from typing import Optional
import pandas as pd
import numpy as np

from src.exception import MyException
from src.constants import DATABASE_NAME
from src.configuration.mongo_db_connection import MongoDBClient

class LoanData:
    """
    A class to export MongoDB records as a pandas DataFrame.
    """
    def __init__(self) -> None:
        """
        Initializes the MongoDB client connection.
        """
        try:
            self.mongo_client = MongoDBClient(database_name = DATABASE_NAME)
        except Exception as e:
            raise MyException(e,sys)
        
    def export_collection_as_dataframe(self, collection_name:str, database_name: Optional[str]=None) -> None:
        """
        Exports an entire MongoDB collection as a pandas DataFrame.

        Parameters:
        ----------
        collection_name : str
            The name of the MongoDB collection to export.
        database_name : Optional[str]
            Name of the database (optional). Defaults to DATABASE_NAME.

        Returns:
        -------
        pd.DataFrame
            DataFrame containing the collection data, with '_id' column removed and 'na' values replaced with NaN.
        """
        try:
            # Access specified collection from the default or specified database
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client.database[database_name][collection_name]
            
            # Convert collection data to pandas dataframe and preprocess
            print("Fetching Data from MongoDB")
            df = pd.DataFrame(list(collection.find()))
            print(f"Data Fetched with length of: {len(df)}")
            if "_id" in df.columns.to_list():
                df = df.drop(columns=['_id'],axis=1)
                
            df.replace({"na":np.nan},inplace=True)
            return df
        except Exception as e:
            raise MyException(e,sys)