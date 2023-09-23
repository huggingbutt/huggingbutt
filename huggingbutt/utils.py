import os
import sys
import zipfile
from pathlib import Path
from typing import List
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from huggingbutt import settings
from huggingbutt.logger_util import get_logger

if int(sys.version.split('.')[1]) > 10:
    import tomllib as toml
else:
    import toml

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
    assert os.path.exists(success_path), f"{success_path} not found."
    Path(success_path).touch()


def agent_download_dest_path(agent_id: int):
    """
    Return the local absolute path of the agent to download
    :param agent_id
    :return:
    """
    return os.path.join(settings.real_cache_path, 'zip', f"agent_{agent_id}.zip")


def local_env_path(user_name, env_name, version):
    return os.path.join(settings.env_path, user_name, env_name, version)


def local_agent_path(agent_id: int):
    return os.path.join(settings.agent_path, str(agent_id))


def compress(files: List[str], desc_path, del_file=False):
    with zipfile.ZipFile(desc_path, 'w') as zip:
        for file in files:
            zip.write(file, arcname=os.path.basename(file))

    if del_file:
        for f in files:
            os.remove(f)



def extract(zip_path, dest_path):
    with zipfile.ZipFile(zip_path, 'r') as zip:
        zip.extractall(dest_path)


def extract_env(user_name, env_name, version):
    zip_file = env_download_dest_path(user_name, env_name, version)
    dest_path = local_env_path(user_name, env_name, version)
    extract(zip_file, dest_path)


def extract_tb_log(path: str) -> pd.DataFrame:
    """
    Extract data from tensorboard log file
    :param path:
    :return:
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    event_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find('tfevents'):
                event_files.append(os.path.join(root, file))

    if len(event_files) == 0:
        raise FileNotFoundError("Event file not found.")
    elif len(event_files) == 1:
        event_file = event_files[0]
    else:
        event_file = max(event_files, key=lambda x: os.path.getctime(x))

    # load the event log file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # get all usable matrices
    metrics = ea.Tags().get('scalars')
    assert len(metrics) > 0, "Not found any metrics."

    data = {}
    # save the step column value
    steps_col = []

    #
    max_length = -1
    for m in metrics:
        values = []
        temp_steps = []
        for d in ea.Scalars(m):
            values.append(d.value)
            temp_steps.append(d.step)

        if len(values) > max_length:
            max_length = len(values)
            # steps_cols will always save the steps with the max length variable.
            steps_col = temp_steps

        data[m.split('/')[-1]] = values

    # Align variable lengths
    for k, v in data.items():
        if len(v) < max_length:
            v.insert(0, 0)

    df = pd.DataFrame(data)
    df.insert(0, 'steps', steps_col)
    return df


def toml_read(path: str) -> dict:
    if toml.__name__ == 'toml':
        with open(file_exists(path), 'r') as f:
            result = toml.load(f)
    elif toml.__name__ == 'tomllib':
        with open(file_exists(path), 'rb') as f:
            result = toml.load(f)
    else:
        raise RuntimeError("toml module is loaded incorrectly.")

    return result


def toml_write(obj: dict, path: str):
    if toml.__name__ == 'toml':
        with open(path, 'w') as f:
            toml.dump(obj, f)
    elif toml.__name__ == 'tomllib':
        with open(path, 'wb') as f:
            toml.dump(obj, f)
    else:
        raise RuntimeError("toml module is loaded incorrectly.")