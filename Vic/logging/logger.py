import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)


    if not logger.handlers:
        logger.addHandler(file_handler)

    return logger