from src.config.config_loader import ConfigLoader
from src.config.config_validator import ConfigValidator

config = ConfigLoader()

validator = ConfigValidator(config)

validator.validate()
