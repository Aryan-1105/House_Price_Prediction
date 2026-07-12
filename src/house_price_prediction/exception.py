# Import the sys module to access exception traceback information
import sys

# Import ModuleType for proper type hinting of the sys module
from types import ModuleType

# Import the custom logger
from src.house_price_prediction.logger import logging


def error_message_detail(
    error: Exception,
    error_detail: ModuleType,
) -> str:
    """
    Generate a detailed error message containing:
    1. File name
    2. Line number
    3. Original error message

    Parameters
    ----------
    error : Exception
        The original exception that occurred.

    error_detail : ModuleType
        The sys module used to retrieve traceback information.

    Returns
    -------
    str
        A formatted error message.
    """

    # Retrieve exception type, exception object, and traceback
    _, _, exc_tb = error_detail.exc_info()

    # If traceback is unavailable, simply return the error message
    if exc_tb is None:
        return f"Error: {str(error)}"

    # Extract the filename where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename

    # Extract the line number where the exception occurred
    line_number = exc_tb.tb_lineno

    # Create a detailed error message
    error_message = (
        f"Error occurred in Python script [{file_name}] "
        f"at line number [{line_number}] "
        f"with error message [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception Class

    This class extends Python's built-in Exception class
    and provides detailed information about the exception,
    including the file name and line number.
    """

    def __init__(
        self,
        error_message: Exception,
        error_detail: ModuleType,
    ) -> None:
        """
        Initialize the custom exception.

        Parameters
        ----------
        error_message : Exception
            The original exception.

        error_detail : ModuleType
            The sys module used to retrieve traceback information.
        """

        # Initialize the parent Exception class
        super().__init__(str(error_message))

        # Generate and store the formatted error message
        self.error_message = error_message_detail(
            error_message,
            error_detail,
        )

    def __str__(self) -> str:
        """
        Return the formatted error message
        whenever the exception is printed.
        """
        return self.error_message