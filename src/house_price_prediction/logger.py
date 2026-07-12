import logging
import os
from datetime import datetime

# ==========================================================
# Create Logs Directory
# ==========================================================

# Root directory of the project
LOG_DIR = "logs"

# Create the logs folder if it doesn't exist
os.makedirs(LOG_DIR, exist_ok=True)

# Create a log file with current date and time
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# ==========================================================
# Configure Logger
# ==========================================================

logging.basicConfig(

    filename=LOG_FILE_PATH,

    format="[ %(asctime)s ] %(levelname)s - %(name)s - %(message)s",

    level=logging.INFO
)