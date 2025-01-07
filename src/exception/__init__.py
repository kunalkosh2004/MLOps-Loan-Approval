import sys
import logging

def error_message_details(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    # Extract Traceback Details
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    line_number = exc_tb.tb_lineno
    error_message = f"Error occured in the file: [{file_name}] at line number: [{line_number}]: {str(error)}"
    
    # logging the error for better interpretetion
    logging.info(error_message)
    
    return error_message

class MyException(Exception):
    """
    Custom exception class for handling errors in the US visa application.
    """
    def __init__(self, error_message:str , error_details: sys):
        """
        Initializes the USvisaException with a detailed error message.

        :param error_message: A string describing the error.
        :param error_detail: The sys module to access traceback details.
        """
        
        # Call the base class constructor with the error msg
        super().__init__(error_message)
        
        # Formatting the detailed error msg using the error_message_detail func
        self.error_message = error_message_details(error_message,error_details)
        
    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message