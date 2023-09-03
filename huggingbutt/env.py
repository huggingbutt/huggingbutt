import sys
import os.path
import re
import subprocess
from huggingbutt.extend_error import EnvNameErrorException
from huggingbutt.utils import local_env_path, get_logger, extract_env, file_exists
from huggingbutt.network import download_env
from huggingbutt.unity_gym_env_small_modified import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from typing import List



if int(sys.version.split('.')[1]) > 10:
    import tomllib as toml
else:
    import toml


logger = get_logger()


def match_env_name(env_name):
    return True if re.match(r'^[a-zA-Z0-9\_]+\/[a-zA-Z0-9\_]+$', env_name) else False


class Env(object):
    def __init__(
            self,
            user_name: str,
            env_name: str,
            version: str,
            env_path: str = None,
            base_port: int = 5000,
            startup_args: List[str] = None
    ):
        """

        :param name:
        :param version:
        :param env_path:
        :param base_port:
        :param startup_args:
        """
        self.user_name: str = user_name
        self.env_name: str = env_name
        self.version: str = version
        self.id: int = -1  # env id on the server
        self.base_port = base_port
        self.startup_args = startup_args
        self.env_path: str = env_path
        self.config: dict = None
        self.exe_file = None
        self.load_config()
        self.gym_env: DummyVecEnv = None

    def load_config(self):
        """
        Extract the environment information.
        """
        if self.env_path is None:
            maybe_path = local_env_path(self.user_name, self.env_name, self.version)
            if os.path.exists(maybe_path):
                self.env_path = maybe_path
            else:
                raise RuntimeError("The parameter env_path is not given, and did not find this environment locally.")

        config_file = os.path.join(self.env_path, 'config.toml')
        if toml.__name__ == 'toml':
            with open(file_exists(config_file), 'r') as f:
                self.config = toml.load(f)
        elif toml.__name__ == 'tomllib':
            with open(file_exists(config_file), 'rb') as f:
                self.config = toml.load(f)
        else:
            raise RuntimeError("The toml module is loaded incorrectly.")

        # check_config_file
        app_config = self.config.get('app', None)
        self.exe_file = os.path.join(self.env_path, app_config.get('exe_file', None))

        if self.exe_file is None:
            raise RuntimeError("Exe file is not defined in configuration file.")
        else:
            if not os.path.isfile(self.exe_file):
                raise RuntimeError(f"{self.exe_file} is not found.")

    def make_gym_env(self):
        """
        Create a wrapped, monitored Unity environment.
        !!! Copied from the official example of mlagents and made a little modification
        """
        if self.gym_env:
            return self.gym_env

        def make_env():  # pylint: disable=C0111
            def _thunk():
                unity_env = UnityEnvironment(self.exe_file, base_port=self.base_port, additional_args=self.startup_args)
                env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
                env = Monitor(env)
                return env

            return _thunk
        self.gym_env = DummyVecEnv([make_env()])
        return self.gym_env

    @classmethod
    def get(cls, env_name, version, startup_args: List[str] = None):
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
        instance = cls(
            user_name=user_name,
            env_name=env_name,
            version=version,
            env_path=local_path,
            startup_args=startup_args
        )
        return instance

    def close(self):
        self.gym_env.close()

    def __del__(self):
        print("Destroy env.")
        if self.gym_env:
            self.close()

    # todo
    def check_config_file(self):
        """
        Check the configuration file for the minimum required parameters.
        """
        pass

    def save(self, to_path: str):
        """
        Save user defined environment.
        """
        pass

    def upload(self):
        """
        Upload this environment to HuggingButt server
        """
        pass



