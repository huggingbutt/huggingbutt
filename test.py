from huggingbutt.env import Env
from huggingbutt.agent import Agent, Agent2
from stable_baselines3.ppo import PPO
from stable_baselines3.common.policies import ActorCriticPolicy

from huggingbutt.agent import get_latest_checkpoint


if __name__ == '__main__':
    env = Env.get("HuggingButt/juggle", 'mac', startup_args=['--time_scale', '5'])
    agent = Agent(
        algorithm=PPO,
        policy=ActorCriticPolicy,
        env=env
    )
    agent.learn(total_timesteps=10000)
    agent.save()
    env.close()





