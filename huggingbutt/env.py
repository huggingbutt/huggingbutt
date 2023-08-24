import sys
import os.path
import re
import subprocess
from huggingbutt.extend_error import EnvNameErrorException
from huggingbutt.utils import local_env_path, get_logger, extract_env, file_exists
from huggingbutt.network import download_env
from huggingbutt.unity_gym_env_small_modified import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from typing import List



if int(sys.version.split('.')[1]) > 10:
    import tomllib as toml
else:
    import toml


logger = get_logger()


# todo...
# def check_env_config(config: dict):
#     for param in settings.env_config_required:
#         if len(param.split('.')) > 1:
#             for p in param.split('.'):

def make_unity_env(env_file, startup_args=None):
    """
    Create a wrapped, monitored Unity environment.
    !!! Copied from the official example of mlagents and made a little modification
    """
    def make_env(startup_args: List[str]=None):  # pylint: disable=C0111
        def _thunk():
          unity_env = UnityEnvironment(env_file, base_port=5000, additional_args=startup_args)
          env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
          env = Monitor(env)
          return env

        return _thunk

    return DummyVecEnv([make_env(startup_args)])


def load_env(local_path: str, startup_args=None):
    config_file = os.path.join(local_path, 'config.toml')
    if toml.__name__ == 'toml':
        with open(file_exists(config_file), 'r') as f:
            config = toml.load(f)
    elif toml.__name__ == 'tomllib':
        with open(file_exists(config_file), 'rb') as f:
            config = toml.load(f)
    else:
        raise RuntimeError("The toml module is loaded incorrectly.")

    # check_env_config(config)

    app_config: dict = config.get('app', None)
    exe_file = app_config.get('exe_file', None)

    if not exe_file:
        raise Exception("Exe file is not defined in configuration file.")

    return make_unity_env(file_exists(os.path.join(local_path, exe_file)), startup_args=startup_args)


def match_env_name(env_name):
    return True if re.match(r'^[a-zA-Z0-9\_]+\/[a-zA-Z0-9\_]+$', env_name) else False


class Env:
    @staticmethod
    def get_gym_env(env_name, version, startup_args=None):
        """
        Returns a gym-type training environment through the specified environment name.
        If there is no such environment locally, it will be downloaded from the remote server.
        :param env_name:
        :param version:
        :param startup_args:
        :return:
        """
        if not match_env_name(env_name):
            raise EnvNameErrorException()
        [user_name, env_name] = env_name.split('/')

        # Check if the environment exists locally
        local_path = local_env_path(user_name, env_name, version)
        if not os.path.exists(local_path):
            download_env(user_name, env_name, version)
            if sys.platform in ('linux', 'darwin'):
                subprocess.run(['chmod', '-R', '755', local_path])

        return load_env(local_path, startup_args=startup_args)

