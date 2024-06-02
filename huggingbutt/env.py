import ast
import sys
import os.path
import re
import subprocess
from typing import Tuple, Dict, Any, SupportsFloat, Optional, List, Union, Callable
import numpy as np
import gymnasium as gym
import tiptoestep as tts
from tiptoestep.action import ContinuousAction
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from .extend_error import EnvNameErrorException
from .utils import local_env_path, get_logger, toml_read, file_exists
from .network import download_env


logger = get_logger()


def match_env_name(env_name):
    return True if re.match(r'^[a-zA-Z0-9\_]+\/[a-zA-Z0-9\_]+$', env_name) else False


def create_action_space(action_config):
    a_space = action_config['space']
    a_shape = action_config['shape']
    if a_space == 'box':
        a_low = np.float32(action_config.get('low', -np.infty))
        a_high = np.float32(action_config.get('high', np.infty))
        return gym.spaces.Box(low=a_low, high=a_high, shape=(a_shape,), dtype=np.float32)
    else:
        raise ValueError('The space of action only supports box type.')


def create_observation_space(observation_config):
    o_space = observation_config['space']
    o_shape = observation_config['shape']

    if not isinstance(o_shape, int):
        raise ValueError('The shape of observation must be an integer.')

    if o_space == 'box':
        o_low = np.float32(observation_config.get('low', -np.infty))
        o_high = np.float32(observation_config.get('high', np.infty))
        return gym.spaces.Box(low=o_low, high=o_high, shape=(o_shape,), dtype=np.float32)
    else:
        raise ValueError('The space of observation only supports box type.')


def create_action(action_config):
    a_type = action_config['type']
    a_shape = action_config['shape']

    if not isinstance(a_shape, int):
        raise ValueError('The shape of action must be an integer.')

    if a_type == 'ContinuousAction':
        action = ContinuousAction(a_shape)
        action_space = create_action_space(action_config)
        return action, action_space
    elif a_type == 'CategoricalAction':
        raise NotImplementedError(f"{a_type} is currently not supported.")
    else:
        raise ValueError(f"{a_type} not supported.")


def load_config(env_path, silent):
    """
    Extract the environment information.
    """
    file_exists(env_path)

    config_file = os.path.join(env_path, 'config.toml')
    config = toml_read(config_file)

    required_keys = ['app', 'action', 'observation', 'function']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"{key} not found in the configuration file.")

    exe_file = os.path.join(env_path, config['app']['exe_file'])
    file_exists(exe_file)

    # Process action information
    action, action_space = create_action(config['action'])

    # Process observation
    observation_space = create_observation_space(config['observation'])

    # Process custom functions information
    fun_file = os.path.join(env_path, config['function']['file'])
    file_exists(fun_file)
    # transform_fun, reward_fun, control_fun = load_custom_functions(fun_file)

    # Load custom function compiled code
    with open(fun_file, 'r') as file:
        function_string = file.read()
    function_ast = ast.parse(function_string)
    custom_fun_code = compile(function_ast, filename="<ast>", mode="exec")

    default_params = {
        "action": action,
        "action_space": action_space,
        "observation_space": observation_space,
        "fun_code": custom_fun_code,
        "exe_file": exe_file
    }

    if silent is True:
        if 'silent_file' in config['app']:
            silent_file = os.path.join(env_path, config['app']['silent_file'])
            default_params['exe_file'] = file_exists(silent_file)
        else:
            raise RuntimeError("Silent model is not supported.")

    return default_params


class Env(object):
    def __init__(
            self,
            user_name: str,
            env_name: str,
            version: str,
            env_path: str = None,
            startup_args = None
    ):
        """

        :param user_name:
        :param env_name:
        :param version:
        :param env_path:
        """
        self.user_name: str = user_name
        self.env_name: str = env_name
        self.version: str = version
        self.id: int = -1  # env id on the server
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

        # todo
        # check_config_file function
        app_config = self.config.get('app', None)
        self.exe_file = os.path.join(self.env_path, app_config.get('exe_file'))

        assert self.exe_file, "Exe file is not defined in configuration file."
        assert os.path.exists(self.exe_file), f"{self.exe_file} not found."

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
            startup_args: List[str] = None):
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
            if sys.platform in ('linux', 'darwin'):  # Adding execute permission if on Linux or MacOS sysetm.
                subprocess.run(['chmod', '-R', '755', local_path])

        instance = cls(
            user_name=user_name,
            env_name=env_name,
            version=version,
            env_path=local_path,
            startup_args=startup_args
        )
        return instance

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


def load_env(env_name, version, silent=False, num=1, **kwargs) -> Union[tts.Env, SubprocVecEnv]:
    """
    Returns a gym-type training environment through the specified environment name.
    If there is no such environment locally, it will be downloaded from the remote server.
    :param env_name:
    :param version:
    :param silent:
    :param num:
    :return:
    """
    if not match_env_name(env_name):
        raise EnvNameErrorException()
    [user_name, env_name] = env_name.split('/')

    # Check if the environment exists locally
    local_path = local_env_path(user_name, env_name, version)
    if not os.path.exists(local_path):
        download_env(user_name, env_name, version)
        if sys.platform in ('linux', 'darwin'):  # Adding execute permission if on Linux or macOS sysetm.
            subprocess.run(['chmod', '-R', '755', local_path])

    # Instancing a tiptoestep environment
    params = load_config(local_path, silent)

    for k, v in kwargs.items():
        params[k] = v

    if num == 1:  # Return a gym.Env instance
        env = tts.Env(**params)
        return Monitor(env)

    def make_env(pid, _params):
        def _init():
            myenv = tts.Env(pid=pid, **_params)
            return Monitor(myenv)

        return _init

    envs = [make_env(pid, params) for pid in range(num)]
    return SubprocVecEnv(envs)


def load_agent():
    pass