import sys
from src.logger import logging

def error_message(error, error_detail: sys):
    _, _, exe_tb = error_detail.exc_info()
    file_name = exe_tb.tb_frame.f_code.co_filename
    message = "error occured in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, exe_tb.tb_lineno, str(error)
    )

    return message
    

class custom_exception(Exception):
    def __init__(self, message, error_detail: sys):
        super().__init__(message)
        self.error_message = error_message(message, error_detail=error_detail)

    def __str__(self):
        return self.error_message