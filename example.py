from huggingbutt import Env, set_access_token
from stable_baselines3.ppo import PPO


if __name__ == '__main__':
    set_access_token("9ba74ddf19b589c0e62a5e773a30a0f2922e3c55")
    env = Env.get_gym_env('HuggingButt/juggle', 'mac', startup_args=['--time_scale', '2'])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("ppo_juggle_2.model")