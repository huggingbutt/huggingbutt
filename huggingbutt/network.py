import os
import requests
from tqdm import tqdm
from huggingbutt import settings
from huggingbutt import utils
from huggingbutt.utils import get_logger, get_access_token, check_token, local_env_path, extract, local_agent_path
from huggingbutt.extend_error import AccessTokenNotFoundException, HubAccessException, VersionNotFoundException



logger = get_logger(__name__)


def get_headers(access_token):
    headers = {
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Authorization": f"Token {access_token}"
    }
    return headers

# todo...
# Get the latest version from a remote server
def get_latest_version(user_name, env_name):
    version = 'latest'
    return version


# Get the md5 of the agent from a remote server
def get_agent_md5(user_name, agent_name, version):
    pass


# Get the md5 of the env from a remote server
def get_env_md5(user_name, env_name, version):
    pass

# Determine whether the downloaded file matches the remote md5
def md5check(file, md5):
    pass


def download(url, to_file_name):
    token = get_access_token()
    if not check_token(token):
        raise AccessTokenNotFoundException()

    headers = get_headers(token)
    response = requests.get(url, headers=headers, stream=True)

    if response.status_code != 200:
        raise HubAccessException(response.text, response.status_code)

    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kb
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    if os.path.exists(to_file_name):
        logger.warning("file {} is exists, will overwrite it.".format(to_file_name))
    try:
        with open(to_file_name, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
    except Exception as ex:
        logger.error(ex)
        exit(-1)
    finally:
        progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logger.error("something went wrong.")


def download_env(user_name: str, env_name: str, version: str):
    """
    Download a env from remote server.
    :param user_name:
    :param env_name:
    :param version:
    :return:
    """

    # todo...
    # Download the latest version by default.
    # if version == 'latest':
    #     version = get_remote_latest_version(user_name, env_name)
    # if version == '':
    #     raise VersionNotFoundException()

    logger.info(f"Download {user_name}/{env_name}:{version}.")
    env_url = f"{settings.hub_url}/download/env/{user_name}/{env_name}_{version}.zip"
    dest_path = utils.env_download_dest_path(user_name, env_name, version)
    download(env_url, dest_path)

    logger.info(f"Extract {user_name}/{env_name}:{version}.")
    extract_path = local_env_path(user_name, env_name, version)
    extract(dest_path, extract_path)


def download_agent(agent_id: int):
    logger.info(f"Download agent {agent_id}.")
    agent_url = f"{settings.hub_url}/download/agent/{agent_id}/"
    dest_path = utils.agent_download_dest_path(agent_id)
    download(agent_url, dest_path)

    logger.info(f"Extract agent {agent_id}")
    extract_path = local_agent_path(agent_id)
    extract(dest_path, extract_path)
