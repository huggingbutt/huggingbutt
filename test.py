import os
import zipfile

from huggingbutt import Env, Agent, set_access_token
from huggingbutt.utils import compress

if __name__ == '__main__':
    set_access_token('08d4db07fe2532a5b629bf7aaf3dc6bc2aba35d3')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '10'])

    policy_kwargs = {
        'normalize_images' : False
    }

    agent = Agent(
        env=env,
        algorithm='PPO',
        policy='MlpPolicy',
        batch_size=256,
        policy_kwargs=policy_kwargs
    )
    agent.learn(total_timesteps=1000)
    agent.save()
    env.close()
    # agent = Agent.get(16, env)
    # print(agent)





