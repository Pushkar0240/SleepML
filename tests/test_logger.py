from src.logging import ProjectLogger

logger = ProjectLogger().get_logger()

logger.info("Project started.")

logger.warning("This is a warning.")

logger.error("Example error message.")

print()

print("Log file created successfully.")
