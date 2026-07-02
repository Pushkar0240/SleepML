from src.logger import logger_factory


logger = logger_factory.get_logger("TEST")
logger.info("Logger initialized successfully.")
logger.warning("This is a warning.")
logger.error("Example error message.")
