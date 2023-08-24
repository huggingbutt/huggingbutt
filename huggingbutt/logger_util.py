import sys
import logging
from huggingbutt import settings
def get_logger(name: str = settings.default_logger_name_) -> logging.Logger:
    """
    Copied From mlagents official project.
    :param name:
    :return:
    """
    _logger = logging.getLogger(name=name)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)
    return _logger