from src.logging import ProjectLogger

logger = ProjectLogger().get_logger()

logger.info("Framework initialized.")

logger.warning("Learning rate is high.")

logger.error("Example training error.")
