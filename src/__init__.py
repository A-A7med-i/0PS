from src.constant.constant import *
import logging
import sys
import os

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt=DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)],
)

logger = logging.getLogger(__name__)
