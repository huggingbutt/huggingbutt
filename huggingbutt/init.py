import os
from huggingbutt import settings
from huggingbutt import utils

logger = utils.get_logger(__name__)


def init():
    if not os.path.exists(settings.real_cache_path):
        logger.info("Create folders.")
        utils.make_dir(settings.real_cache_path)
        utils.make_dir(settings.zip_path)
        utils.make_dir(settings.env_path)
        utils.make_dir(settings.agent_path)
        utils.make_dir(settings.downloaded_path)
        utils.make_dir(settings.downloaded_env_path)
        utils.make_dir(settings.downloaded_agent_path)







