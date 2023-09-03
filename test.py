from huggingbutt.env import Env
from huggingbutt.agent import Agent
from stable_baselines3.ppo import PPO

if __name__ == '__main__':
    env = Env.get("HuggingButt/juggle", 'win', startup_args=['--time_scale', '10'])
    model = PPO('MlpPolicy', env.make_gym_env(), verbose=1)
    agent = Agent(model, env)
    agent.learn(total_timesteps=80000)
    env.close()



