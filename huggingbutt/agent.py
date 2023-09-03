from abc import ABC

import pandas as pd
from tensorboard.backend.event_processing import event_accumulator
from huggingbutt.env import Env
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.type_aliases import MaybeCallback
def extract_log(path: str, to_path: str=None):
    ea = event_accumulator.EventAccumulator(
        'head_juggle_training_log/PPO_2/events.out.tfevents.1693655469.DESKTOP-NRBS60M.9788.0')



def extract_tb_log(path: str) -> pd.DataFrame:
    return None


class TrainingEndCallBack(BaseCallback, ABC):
    def __int__(self, tb_log_dir: str, to_path: str):
        self.tb_log_dir = tb_log_dir
        self.to_path = to_path

    def _on_training_end(self) -> None:
        df = extract_tb_log(self.tb_log_dir)
        df.to_pickle(self.to_path)

    def _on_step(self):
        return True


class Agent:
    def __init__(self, model: BaseAlgorithm, env: Env, log_dir: str = None):
        self.env: Env = env
        self.id: int = -1  # agent id on the server
        if model.tensorboard_log is None:
            if log_dir is None:
                model.tensorboard_log = f"tensorboard_log_{model.__class__.__name__}_sb3"
            else:
                model.tensorboard_log = log_dir
        self.model = model
        self.save_path: str = None

    def learn(
            self,
            total_timesteps: int,
            save_freq: int = 2048,
            save_path: str = None,
            log_interval: int = 100,
            tb_log_name: str = "run",
            reset_num_timesteps: bool = True,
            progress_bar: bool = False,
    ):

        if save_path is None:
            self.save_path = f"{self.model.__class__.__name__}_sb3"

        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=self.save_path,
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


