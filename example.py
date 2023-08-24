from huggingbutt import Env, set_access_token
from stable_baselines3.ppo import PPO
ACCESS_TOKEN = 'YOUR_TOKEN'

if __name__ == '__main__':
    set_access_token(ACCESS_TOKEN)
    env = Env.get_gym_env('HuggingButt/juggle', 'mac', startup_args=['--time_scale', '2'])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save('ppo_juggle.model')