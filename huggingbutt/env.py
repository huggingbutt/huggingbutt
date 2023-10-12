import sys
import os.path
import re
import subprocess
from typing import Tuple, Dict, Any, SupportsFloat, Optional
from huggingbutt.extend_error import EnvNameErrorException
from huggingbutt.utils import local_env_path, get_logger, toml_read
from huggingbutt.network import download_env
from huggingbutt.unity_gym_env_small_modified import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv, VecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.core import ActType, ObsType, Env
from typing import List

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
            base_port: int = 5005,
            work_id: int = 0,
            startup_args: List[str] = None
    ):
        """

        :param user_name:
        :param env_name:
        :param version:
        :param env_path:
        :param base_port:
        :param work_id:
        :param startup_args:
        """
        self.user_name: str = user_name
        self.env_name: str = env_name
        self.version: str = version
        self.id: int = -1  # env id on the server
        self.base_port = base_port
        self.work_id = work_id
        self.startup_args = startup_args
        self.env_path: str = env_path
        self.config: dict = {}
        self.exe_file: Optional[str] = None
        self.gym_env: Optional[Env] = None
        self.load_config()

    def load_config(self):
        """
        Extract the environment information.
        """
        if self.env_path is None:
            maybe_path = local_env_path(self.user_name, self.env_name, self.version)
            assert os.path.exists(maybe_path), "The parameter env_path is not given, " \
                                               "and did not find this environment locally."
            self.env_path = maybe_path

        config_file = os.path.join(self.env_path, 'config.toml')
        self.config = toml_read(config_file)

        # need to do check_config_file
        app_config = self.config.get('app', None)
        self.exe_file = os.path.join(self.env_path, app_config.get('exe_file'))

        assert self.exe_file, "Exe file is not defined in configuration file."
        assert os.path.exists(self.exe_file), f"{self.exe_file} is not found."

    def make_gym_env(self):
        """
        Create a wrapped, monitored Unity environment.
        !!! Copied from the official example of mlagents and made a little modification
        """
        if self.gym_env:
            return self.gym_env

        def make_env():  # pylint: disable=C0111
            def _thunk():
                unity_env = UnityEnvironment(self.exe_file,
                                             base_port=self.base_port,
                                             additional_args=self.startup_args,
                                             worker_id=self.work_id)
                env = UnityToGymWrapper(unity_env, uint8_visual=False, allow_multiple_obs=False)
                env = Monitor(env)
                return env

            return _thunk
        self.gym_env = DummyVecEnv([make_env()])
        return self.gym_env

    def agent_list(self):
        """
        List all agents of this environment.
        :return:
        """
        pass

    def get_agent(self, agent_id: int):
        """
        Instance an agent for this environment by agent_id.
        :param agent_id:
        :return:
        """
        pass

    @classmethod
    def get(cls,
            env_name,
            version,
            base_port: int = 5005,
            work_id: int = 0,
            startup_args: List[str] = None):
        """
        Returns a gym-type training environment through the specified environment name.
        If there is no such environment locally, it will be downloaded from the remote server.
        :param env_name:
        :param version:
        :param base_port:
        :param work_id:
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
            base_port=base_port,
            work_id=work_id,
            startup_args=startup_args
        )
        return instance

    def reset(self, **kwargs) -> Tuple[ObsType, Dict[str, Any]]:
        if self.gym_env is None:
            self.make_gym_env()
        return self.gym_env.reset(**kwargs)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, Dict[str, Any]]:
        if self.gym_env is None:
            raise RuntimeError('You need to execute function reset() first.')
        return self.gym_env.step(action)

    def close(self):
        self.gym_env.close()

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



