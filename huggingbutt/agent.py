import os
import pathlib
from typing import Union, Type, Any, List, TypeVar, Dict
from functools import cmp_to_key
from abc import ABC
from huggingbutt.utils import extract_tb_log
from huggingbutt.env import Env
from huggingbutt.utils import file_exists, get_logger, toml_write
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO, A2C, DDPG, TD3, DQN, SAC

usable_algorithms = {
    'PPO': PPO,
    'A2C': A2C,
    'DDPG': DDPG,
    'TD3': TD3,
    'DQN': DQN,
    'SAC': SAC
}

logger = get_logger(__name__)


def get_latest_checkpoint(path: str) -> (str, int):
    """
    Get the latest checkpoint file.
    This is used to obtain the latest model file to avoid repeated saving.
    :param path:
    :return:
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    assert os.path.exists(path), f"{path} is not exists."

    files = []
    for f in os.listdir(file_exists(path)):
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.zip'):
            files.append(os.path.join(path, f))

    if len(files) == 0:
        raise FileNotFoundError('Checkpoint files found.')
    elif len(files) == 1:
        latest_file = files[0]
    else:
        latest_file = max(files, key=lambda x: os.path.getctime(x))

    try:
        time_steps = int(os.path.basename(latest_file).split('_')[2])
    except:
        time_steps = -1

    return latest_file, time_steps


class TrainingEndCallBack(BaseCallback, ABC):
    """
    This class will be executed after the model training is finished.
    """
    def __init__(self, tb_log_dir: str, to_path: str):
        super().__init__()
        self.tb_log_dir = tb_log_dir
        self.to_path = to_path

    def _on_training_end(self) -> None:
        df = extract_tb_log(self.tb_log_dir)
        df.to_csv(self.to_path, index=False)
        logger.info(f"Log data is stored in {self.to_path}, you can upload it to HuggingButt server.")

    def _on_step(self):
        """
        return True by default.
        :return:
        """
        return True


def check_algorithm_class(cls: Type[BaseAlgorithm], candidates: Dict[str, Type[BaseAlgorithm]]):
    """
    To check parameter cls: BaseAlgorithm passed by user.
    :param cls:
    :param candidates:
    :return:
    """
    for k,v in candidates.items():
        if cls is v:
            return cls
    raise ValueError(f"Type {cls} is not supported.")


def _get_algo_from_name(algo_name: str) -> Type[BaseAlgorithm]:
    if algo_name in usable_algorithms:
        return usable_algorithms[algo_name]
    else:
        raise ValueError(f"Algorithm {algo_name} unknown. Usable algorithms: {usable_algorithms.keys()}")


def get_next_path(path: str):
    dir_name = path
    count = 1
    while True:
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
            break
        else:
            dir_name = f"{path}_{count}"
            count += 1
    return dir_name


class Agent:
    def __init__(
            self,
            algorithm: Union[str, Type[BaseAlgorithm]],
            policy: Union[str, Type[BasePolicy]],
            env: Env = None,
            save_path: str = None,
            **kwargs,
    ):
        """

        :param algorithm:
        :param policy:
        :param env:
        :param train_info_path:
        :param kwargs:
        """
        # An instance of algorithm class.
        # It will be instantiated when the learn() function is executed.
        self.model = None
        if isinstance(algorithm, str):
            self.algorithm_class = _get_algo_from_name(algorithm)
        else:
            self.algorithm_class = check_algorithm_class(algorithm, usable_algorithms)
        self.policy_class = policy
        self.env = env
        self.id: int = -1  # agent id on the server

        if save_path is None:
            curr_path = os.getcwd()
            self.save_path = get_next_path(os.path.join(curr_path, f"{self.env.env_name}_agent"))
        else:
            self.save_path = save_path

        # Get the parameters for initializing the algorithm class
        self.init_kv = dict()
        if kwargs:
            try:
                usable_kv = self.algorithm_class.__init__.__code__.co_varnames
                for k in kwargs:
                    if k in usable_kv:
                        self.init_kv[k] = kwargs.get(k)
            except:
                logger.warning(f"Parsing kwargs error. All additional parameters ignored.")
                self.init_kv = dict()

        self.save_freq: int = -1
        self.total_timesteps: int = -1
        self.trained_steps: int = -1

        self.tb_log_dir = os.path.join(
            self.save_path,
            f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_sb3_tb_log")
        # tb_log_name is the subdirectory in tensorboard log directory
        self.tb_log_name = self.algorithm_class.__name__
        self.train_log_upload = os.path.join(
            self.save_path,
            f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_sb3_log_up.csv")
        self.checkpoint_dir = os.path.join(
            self.save_path,
            f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_sb3_model")
        self.checkpoint_name_prefix = f"{self.algorithm_class.__name__.lower()}_model"

    def learn(
            self,
            total_timesteps: int,
            save_freq: int = -1,
            log_interval: int = 1,
            verbose: int = 0,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        """

        :param total_timesteps:
        :param save_freq:
        :param log_interval:
        :param verbose:
        :param reset_num_timesteps:
        :param progress_bar:
        :return:
        """
        assert self.env is not None, "env is None"

        self.total_timesteps = total_timesteps  # this is your target total time steps.

        self.model = self.algorithm_class(
            self.policy_class,
            self.env.make_gym_env(),
            tensorboard_log=self.tb_log_dir,
            verbose=verbose,
            **self.init_kv
        )

        # persist model information
        self.model_param_path = os.path.join(
            self.save_path,
            f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_{self.model.policy_class.__name__}_param.toml"
        )

        if save_freq == -1:
            self.save_freq = int(total_timesteps / 5)

        checkpoint_callback = CheckpointCallback(
            name_prefix=self.checkpoint_name_prefix,
            save_freq=self.save_freq,
            save_path=self.checkpoint_dir
        )

        end_callback = TrainingEndCallBack(self.tb_log_dir, self.train_log_upload)

        callbacks = [checkpoint_callback, end_callback]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            tb_log_name=self.tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )

    def save(self):
        path = os.path.join(
            self.save_path,
            f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_sb3_model.zip"
        )

        # get latest checkpoint
        latest_checkpoint, self.trained_steps = get_latest_checkpoint(self.checkpoint_dir)

        if self.trained_steps >= self.total_timesteps:
            # Trained steps of the latest checkpoint file is reaches the target total time steps.
            logger.info(f"Save the latest checkpoint model. Trained Steps {self.trained_steps}.")
            os.rename(latest_checkpoint, path)
        else:
            self.model.save(path)

        toml_write(vars(self.model), self.model_param_path)

    @classmethod
    def get(cls, agent_id: int, env: Env = None):
        print(cls)


