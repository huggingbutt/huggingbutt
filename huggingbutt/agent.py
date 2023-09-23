import os
from typing import Union, Type, Any, Dict, Optional, Tuple
from abc import ABC
import numpy as np
import pandas as pd

from huggingbutt.utils import extract_tb_log
from huggingbutt.env import Env
from huggingbutt.utils import file_exists, get_logger, toml_write, local_agent_path, compress, toml_read
from huggingbutt.network import download_agent
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

agent_keys = [
    'device',
    'verbose',
    'num_timesteps',
    '_total_timesteps',
    '_num_timesteps_at_start',
    'learning_rate',
    'use_sde',
    'sde_sample_freq',
    '_n_updates',
    'observation_space',
    'action_space',
    'n_envs',
    'n_steps',
    'gamma',
    'gae_lambda',
    'ent_coef',
    'vf_coef',
    'max_grad_norm',
    'batch_size',
    'n_epochs',
    'normalize_advantage',
    'policy_kwargs'
]

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


def get_algo_from_name(algo_name: str) -> Type[BaseAlgorithm]:
    if algo_name in usable_algorithms:
        return usable_algorithms[algo_name]
    else:
        raise ValueError(f"Algorithm {algo_name} unknown. Usable algorithms: {usable_algorithms.keys()}")


def get_next_path(path: str):
    dir_name = path
    count = 1
    while True:
        if not os.path.exists(dir_name):
            break
        else:
            dir_name = f"{path}_{count}"
            count += 1
    return dir_name


class Agent:
    def __init__(
            self,
            env: Env,
            algorithm: Union[str, Type[BaseAlgorithm]],
            policy: Union[str, Type[BasePolicy]],
            policy_kwargs: Optional[Dict[str, Any]] = None,
            save_path: str = None,
            pretrained: bool = False,
            model_file: str = None,
            agent_id: int = None,
            **kwargs,
    ):
        """

        :param env:
        :param algorithm:
        :param policy:
        :param policy_kwargs:
        :param save_path:
        :param pretrained:
        :param model_file:
        :param agent_id:
        :param kwargs:
        """
        # An instance of algorithm class.
        # It will be instantiated when the learn() function is executed.

        if isinstance(algorithm, str):
            self.algorithm_class = get_algo_from_name(algorithm)
        else:
            self.algorithm_class = check_algorithm_class(algorithm, usable_algorithms)

        self.policy_class = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.env = env
        self.agent_id: int = -1  # agent id on the server

        self.pretrained = pretrained

        if pretrained:
            self.model = self.algorithm_class.load(model_file, env.make_gym_env())
            self.agent_id = agent_id
        else:
            self.model: Optional[BaseAlgorithm] = None

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
            except AttributeError:
                logger.warning(f"Parsing kwargs error. All additional parameters ignored.")
                self.init_kv = dict()

        self.save_freq: int = -1
        self.total_timesteps: int = -1
        self.trained_steps: int = -1
        self.model_param_path: Optional[str] = None

        # tensorboard log dir
        self.tb_log_dir = os.path.join(self.save_path, 'tb_log')
        # tb_log_name is the subdirectory in tensorboard log directory
        self.tb_log_name = self.algorithm_class.__name__
        # training metrics csv file
        self.train_log_upload = os.path.join(self.save_path, 'log_data.csv')

        # checkpoint dir
        self.checkpoint_dir = os.path.join(self.save_path, 'checkpoints')
        self.checkpoint_name_prefix = self.algorithm_class.__name__.lower()

        self.model_file_name = 'model.zip'
        self.model_ful_path = os.path.join(self.save_path, self.model_file_name)

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

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        if not self.pretrained:
            self.model = self.algorithm_class(
                self.policy_class,
                self.env.make_gym_env(),
                tensorboard_log=self.tb_log_dir,
                verbose=verbose,
                policy_kwargs=self.policy_kwargs,
                **self.init_kv
            )
        else:
            self.model.tensorboard_log = self.tb_log_dir

        # persist model information
        self.model_param_path = os.path.join(self.save_path, f"config.toml")

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

    def resume(self):
        """
        todo
        :return:
        """
        pass

    def save(self):
        # Save important training information to file
        agent_dict = dict()
        agent_dict['algorithm_class'] = self.algorithm_class.__name__
        agent_dict['policy'] = self.policy_class
        agent_dict['env'] = f"{self.env.user_name}/{self.env.env_name}@{self.env.version}"
        agent_dict['model_file'] = self.model_file_name
        for k in agent_keys:
            agent_dict[k] = self.model.__dict__.get(k)

        if self.pretrained:
            agent_dict['base_agent'] = self.agent_id

        try:
            df = pd.read_csv(self.train_log_upload)
            latest_metrics = df.iloc[-1].to_dict()
            agent_dict['ep_rew_mean'] = latest_metrics.get('ep_rew_mean')
        except (FileNotFoundError, IndexError):
            logger.warning('Get ep_rew_mean error.')

        # lr_schedule will be handled in subsequent versions

        toml_write(agent_dict, self.model_param_path)

        # clearn model,
        self.model.ep_info_buffer.clear()
        self.model.tensorboard_log = 'tb_log'
        self.model._last_obs = np.array([])
        self.model._last_episode_starts = np.array([])

        self.model.save(self.model_ful_path)

        # Create zip file that needs to be uploaded
        compress([self.model_ful_path, self.model_param_path], os.path.join(self.save_path, f"hb_{self.env.env_name}_{self.algorithm_class.__name__.lower()}.zip"), del_file=True)

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if self.model is None:
            raise RuntimeError("model is None.")

        return self.model.predict(observation, state, episode_start, deterministic)

    @classmethod
    def get(cls, agent_id: int, env: Env):
        local_path = local_agent_path(agent_id)
        if not os.path.exists(local_path):
            download_agent(agent_id)

        assert os.path.isfile(os.path.join(local_path, 'config.toml')), 'config.toml file not found.'
        config = toml_read(os.path.join(local_path, 'config.toml'))

        try:
            algorithm_cls = config['algorithm_class']
            policy_cls = config['policy']
            model_file = config['model_file']
            policy_kwargs = config['policy_kwargs']
        except KeyError:
            raise RuntimeError('Agent config.toml is incomplete!!')

        full_name = os.path.join(local_path, model_file)

        instance = cls(
            env=env,
            algorithm=algorithm_cls,
            policy=policy_cls,
            policy_kwargs=policy_kwargs,
            pretrained=True,
            model_file=full_name,
            agent_id=agent_id
        )

        return instance






