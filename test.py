import os
import zipfile

from huggingbutt import Env, Agent, set_access_token
from huggingbutt.utils import compress

if __name__ == '__main__':
    set_access_token('08d4db07fe2532a5b629bf7aaf3dc6bc2aba35d3')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '10'])

    agent = Agent(
        algorithm='PPO',
        policy='MlpPolicy',
        env=env,
        batch_size=256
    )



    agent.learn(total_timesteps=1000)

    agent.save()
    env.close()
    # agent = Agent.get(12)


    # cur_path = os.getcwd()
    # toml_file = os.path.join(cur_path, 'juggle_agent_9', 'juggle_ppo_ActorCriticPolicy_param.toml')
    # model_file = os.path.join(cur_path, 'juggle_agent_9', 'juggle_ppo_sb3_model.zip')
    # with zipfile.ZipFile(os.path.join(cur_path, 'mymodel.zip'), 'w') as zip:
    #     for file in [toml_file, model_file]:
    #         zip.write(file, arcname=os.path.basename(file))




