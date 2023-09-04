from typing import Union, Type, Any, List, TypeVar
from abc import ABC
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from huggingbutt.env import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO, A2C, DDPG, TD3, DQN, SAC


SelfPPO = TypeVar('SelfPPO', bound=PPO)

def extract_log(path: str, to_path: str=None):
    ea = event_accumulator.EventAccumulator('head_juggle_training_log/PPO_2/events.out.tfevents.1693655469.DESKTOP-NRBS60M.9788.0')


def extract_tb_log(path: str) -> pd.DataFrame:
    """
    Extracting tensorboard log data.
    :param path:
    :return:
    """
    return None


class TrainingEndCallBack(BaseCallback, ABC):
    """
    This call back will be run when training is over.
    """
    def __int__(self, tb_log_dir: str, to_path: str):
        self.tb_log_dir = tb_log_dir
        self.to_path = to_path

    def _on_training_end(self) -> None:
        df = extract_tb_log(self.tb_log_dir)
        df.to_pickle(self.to_path)

    def _on_step(self):
        return True

candidate_algorithms = [

]

def check_algorithm_class(cls: Type[BaseAlgorithm], candidates: List[Type[BaseAlgorithm]]):
    for ca in candidates:
        if isinstance(cls, ca):
            return ca


class Agent:
    def __init__(
            self,
            algorithm: Type[BaseAlgorithm],
            policy: Type[BasePolicy],
            env: Env,
            **kwargs,
    ):

        self.algorithm_class = algorithm
        self.policy_class = policy
        self.env = env
        self.id: int = -1  # agent id on the server

        # Get the parameters for initializing the algorithm class
        self.init_kv = dict()
        if kwargs:
            try:
                usable_kv = algorithm.__init__.__code__.co_varnames
                for k in kwargs:
                    if k in usable_kv:
                        self.init_kv[k] = kwargs.get(k)
            except:
                self.init_kv = dict()

        self.model = None

        # learning parameters
        self.tb_log_dir = ''

    def learn(
            self,
            total_timesteps: int,
            save_freq: int = -1,
            tb_log_dir: str = None,
            tb_log_name: str = 'run',
            checkpoint_dir: str = None,
            log_interval: int = 100,
            verbose: int = 1,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False

    ):
        if tb_log_dir is None:
            self.tb_log_dir = f"{self.algorithm_class.__name__}_sb3_{self.env.env_name}_tb_log"
        else:
            self.tb_log_dir = tb_log_dir

        # instance
        self.model = self.algorithm_class(
            self.policy_class,
            self.env.make_gym_env(),
            tensorboard_log=self.tb_log_dir,
            verbose=verbose,
            **self.init_kv
        )

        if checkpoint_dir is None:
            self.checkpoint_dir = f"{self.algorithm_class.__name__}_sb3_{self.env.env_name}_model"
        else:
            self.checkpoint_dir = checkpoint_dir

        if save_freq == -1:
            self.save_freq = int(total_timesteps / 5)

        checkpoint_callback = CheckpointCallback(
            save_freq=self.save_freq,
            save_path=self.checkpoint_dir,
        )

        callbacks = [checkpoint_callback, TrainingEndCallBack()]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )

    @classmethod
    def get_pretrained(cls):
        print(cls)


