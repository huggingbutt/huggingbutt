import os
import pathlib
from typing import Union, Type, Any, List, TypeVar
from functools import cmp_to_key
from abc import ABC
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from huggingbutt.env import Env
from huggingbutt.utils import file_exists, get_logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3 import PPO, A2C, DDPG, TD3, DQN, SAC
from stable_baselines3.common.policies import ActorCriticPolicy

usable_algorithms = [PPO, A2C, DDPG, TD3, DQN, SAC]

logger = get_logger(__name__)

# def extract_tb_log(path: str, to_path: str=None):
#     ea = event_accumulator.EventAccumulator('head_juggle_training_log/PPO_2/events.out.tfevents.1693655469.DESKTOP-NRBS60M.9788.0')


def extract_tb_log(path: str) -> pd.DataFrame:
    """
    Extract data from tensorboard log files for upload to the server.
    :param path:
    :return:
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    event_file = ''

    for root, dirs, files in os.walk(path):
        for file in files:
            if file.find('tfevents'):
                event_file = os.path.join(root, file)
                break

    if not event_file:
        raise RuntimeError("Not found tensorboard events log file.")

    # load the event log file
    ea = event_accumulator.EventAccumulator(event_file)
    ea.Reload()

    # get all usable matrices
    metrics = ea.Tags().get('scalars')
    if len(metrics) < 1:
        raise RuntimeError("Not found any metrics.")

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
        data[m] = values

    # Align variable lengths
    for k, v in data.items():
        if len(v) < max_length:
            v.insert(0, 0)

    df = pd.DataFrame(data)
    df.insert(0, 'steps', steps_col)
    return df


def timesteps_sort(a, b):
    """
    Checkpoint file is like xx_xx_{timesteps:int}_steps.zip, this cmp function is used to sorted function.
    :param a:
    :param b:
    :return:
    """
    a = int(a.split('_')[2])
    b = int(b.split('_')[2])

    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


def get_latest_checkpoint(path: str) -> (str, int):
    """
    Get the latest checkpoint file base on the time steps.
    This is mainly used to obtain the latest model file to avoid repeated saving.
    :param path:
    :return:
    """
    if not os.path.isabs(path):
        path = os.path.abspath(path)

    files = []
    for f in os.listdir(file_exists(path)):
        if os.path.isfile(os.path.join(path, f)) and f.endswith('.zip'):
            files.append(f)

    if len(files) < 1:
        raise RuntimeError("Checkpoint model files has not saved.")
    elif len(files) == 1:
        latest_file = files[0]
    else:
        sorted(files, key=cmp_to_key(timesteps_sort))
        latest_file = files[0]
    try:
        time_steps = int(latest_file.split('_')[2])
    except:
        time_steps = -1

    return latest_file, time_steps


class TrainingEndCallBack(BaseCallback, ABC):
    """
    It will be executed after the model training is finished.
    """
    def __init__(self, tb_log_dir: str, to_path: str):
        super().__init__()
        self.tb_log_dir = tb_log_dir
        self.to_path = to_path

    def _on_training_end(self) -> None:
        # df = extract_tb_log(self.tb_log_dir)
        # df.to_pickle(self.to_path)
        print("Training Over.")

    def _on_step(self):
        return True


def check_algorithm_class(cls: Type[BaseAlgorithm], candidates: List[Type[BaseAlgorithm]]):
    if cls in candidates:
        return cls
    raise RuntimeError(f"Type {cls} is not supported.")


class Agent2:
    def __init__(self, env: Env):
        self.env = env
        self.model = PPO(ActorCriticPolicy, env.make_gym_env(), verbose=1, tensorboard_log='tb_log')

    def learn(self):
        checkpoint_callback = CheckpointCallback(
            save_freq=2048,
            save_path='PPO_model'
        )

        callbacks = [checkpoint_callback, TrainingEndCallBack()]

        self.model.learn(total_timesteps=20000, callback=checkpoint_callback)


class Agent:
    def __init__(
            self,
            algorithm: Type[BaseAlgorithm],
            policy: Type[BasePolicy],
            env: Env,
            **kwargs,
    ):
        self.algorithm_class = check_algorithm_class(algorithm, usable_algorithms)
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

        self.model: BaseAlgorithm = None

        # learning parameters
        self.tb_log_dir: str = ''
        self.tb_log_name: str = ''
        self.checkpoint_dir: str = ''
        self.checkpoint_name_prefix = ''
        self.save_freq: int = -1
        self.total_timesteps: int = -1
        self.trained_steps: int = -1
        self.train_log_upload: str = ''


    def learn(
            self,
            total_timesteps: int,
            save_freq: int = -1,
            tb_log_dir: str = None,
            log_for_update: str = None,
            checkpoint_dir: str = None,
            checkpoint_name_prefix: str = None,
            log_interval: int = 1,
            verbose: int = 0,
            reset_num_timesteps: bool = True,
            progress_bar: bool = False
    ):
        self.total_timesteps = total_timesteps

        if tb_log_dir is None:
            self.tb_log_dir = f"{self.algorithm_class.__name__}_sb3_{self.env.env_name}_tb_log"
        else:
            self.tb_log_dir = tb_log_dir

        # tb_log_name is the event log folder name
        self.tb_log_name = self.algorithm_class.__name__

        # self.algorithm_class(self.policy_class, self.env.make_gym_env(), verbose=1, tensorboard_log='tb_log')
        # instance
        self.model = self.algorithm_class(
            self.policy_class,
            self.env.make_gym_env(),
            tensorboard_log=self.tb_log_dir,
            verbose=verbose,
            **self.init_kv
        )

        if checkpoint_dir is None:
            self.checkpoint_dir = f"{self.env.env_name}_sb3_model"
        else:
            self.checkpoint_dir = checkpoint_dir

        if save_freq == -1:
            self.save_freq = int(total_timesteps / 5)

        if checkpoint_name_prefix is None:
            self.checkpoint_name_prefix = f"{self.algorithm_class.__name__.lower()}_model"

        checkpoint_callback = CheckpointCallback(
            name_prefix=self.checkpoint_name_prefix,
            save_freq=self.save_freq,
            save_path=self.checkpoint_dir
        )

        end_callback = TrainingEndCallBack(self.tb_log_dir, )

        callbacks = [checkpoint_callback, ]

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=log_interval,
            tb_log_name=self.tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar
        )

    def save(self, path: str = None):
        if path is None:
            path = f"{self.env.env_name}_{self.algorithm_class.__name__.lower()}_sb3_model.zip"
        # get latest checkpoint
        latest_checkpoint, self.trained_steps = get_latest_checkpoint(self.checkpoint_dir)

        if self.trained_steps >= self.total_timesteps:
            # Training steps of the latest checkpoint file is reaches the target total time steps.
            logger.info(f"Save the latest checkpoint model. Trained Steps {self.trained_steps}.")
            pathlib.Path.rename(os.path.join(self.tb_log_dir, latest_checkpoint), path)
        else:
            # save model
            self.model.save(path)



    @classmethod
    def get_pretrained(cls):
        print(cls)


