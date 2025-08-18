import logging

import os
import re

def extract_customer_id(file_path: str) -> str:
    parts = file_path.replace("\\", "/").split("/")
    uuid_pattern = re.compile(r"^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$", re.I)

    for part in parts:
        if uuid_pattern.match(part):
            return part
    return None  # or raise ValueError("Customer ID not found in path.")



def get_logger(name: str, log_file: str, console: bool = True) -> logging.Logger:
    """Create and configure a logger with file and optional console output"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Disable propagation to root logger
    
    # Avoid duplicate handlers
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    
    return logger