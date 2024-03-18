import logging


def set_log_level(level_name):
    """
    Change the debug level for the logger and its handlers.
    level_name (str): The level name as a string, e.g., 'DEBUG', 'INFO', etc.
    """
    level = getattr(logging, level_name.upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    # Set the logger level
    logger.setLevel(level)

    # Set the level for all handlers of the logger
    for handler in logger.handlers:
        handler.setLevel(level)


# Create a logger object
logger = logging.getLogger("global_logger")
logger.setLevel(logging.DEBUG)  # Set logging level

# Create a console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# Create a formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)
