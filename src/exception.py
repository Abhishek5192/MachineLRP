import sys
from .logger import logging


def error_message_details(error,error_detail:sys): # type: ignore
    _,_,exc_traceback=error_detail.exc_info()
    filename = exc_traceback.tb_frame.f_code.co_filename # type: ignore
    line_number = exc_traceback.tb_lineno # type: ignore
    return f"Error occured in python script name[{filename}] at line number [{line_number}] error message format[{str(error)}]"

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys): # type: ignore
        super().__init__(error_message)
        self.error_message = error_message_details(error_message,error_detail)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        a = 1/0
    except Exception as e:
        logging.error("Division by zero error")
        raise CustomException(e,sys)