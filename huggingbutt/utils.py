"""
function descriptions
zhengliang zhang @ shizhuang.inc
"""
import sys
import os
import zipfile
from pathlib import Path
from huggingbutt import settings
from huggingbutt.logger_util import get_logger


logger = get_logger()


def get_access_token():
    return os.environ.get("AGENTHUB_ACCESS_TOKEN")


def set_access_token(token):
    os.environ["AGENTHUB_ACCESS_TOKEN"] = token


def check_path(path):
    return os.path.exists(path)

def file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return path


def make_dir(path):
    if not check_path(path):
        os.makedirs(path)


def check_token(token):
    if token is None or token == "":
        return False
    return True


def safe_file_path(path):
    """
    Make sure the parent directory of the file already exists, if not, create it.
    :param path:
    :return:
    """
    if os.path.exists(path):
        raise FileExistsError("Target File is exists")
    parent_path = os.path.dirname(path)
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    return path


def env_download_dest_path(user_name: str, env_name: str, version: str):
    """
    Return the local absolute path of the env to download
    :param user_name:
    :param env_name:
    :param version:
    :return:
    """
    return os.path.join(settings.real_cache_path, 'zip', f"{user_name}@{env_name}@{version}.zip")


def succ_env_path(user_name, env_name, version):
    return os.path.join(settings.downloaded_env_path, f"{user_name}@{env_name}@{version}")


def touch_succ_env(user_name, env_name, version):
    success_path = succ_env_path(user_name, env_name, version)
    if not os.path.exists(success_path):
        Path(success_path).touch()
    else:
        raise FileExistsError("File is exists.")


def agent_download_dest_path(user_name: str, agent_name: str, version: str, env_user_name: str, env_name: str):
    """
    Return the local absolute path of the agent to download
    :param user_name:
    :param agent_name:
    :param version:
    :return:
    """
    desc_files = os.path.join(settings.real_cache_path, 'zip', f"{user_name}@{agent_name}@{version}@{env_user_name}@{env_name}.zip")


def local_env_path(user_name, env_name, version):
    return os.path.join(settings.env_path, user_name, env_name, version)




def extract(zip_path, dest_path):
    with zipfile.ZipFile(zip_path, "r") as zip:
        zip.extractall(dest_path)


def extract_env(user_name, env_name, version):
    zip_file = env_download_dest_path(user_name, env_name, version)
    dest_path = local_env_path(user_name, env_name, version)
    extract(zip_file, dest_path)
    # todo ...
    # Add executable permission to the env file.
    # ***.app/Contents/MacOS/***

