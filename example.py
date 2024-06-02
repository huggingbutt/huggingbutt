import numpy as np
from huggingbutt import set_access_token, load_env, load_agent

HB_TOKEN = "YOUR_TOKEN"
set_access_token(HB_TOKEN)



if __name__ == '__main__':
    env = load_env("huggingbutt/roller_ball", "mac", silent=True, num=1, time_scale=20)
    # Training
    # model = PPO(
    #     "MlpPolicy",
    #     env,
    #     verbose=1,
    #     n_steps=1024,
    #     tensorboard_log="./logs")
    #
    # model.learn(total_timesteps=1_000_000)
    # model.save(f"roller_ball.zip")
    #
    # env.close()

    # Inference
    agent = load_agent(4)
    obs, info = env.reset()
    steps = []
    rewards = []
    prev_i = 0
    epoch_reward = 0
    for i in range(1_000):
        act, _status_ = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(act)
        epoch_reward += reward

        if terminated:
            obs, info = env.reset()
            # Statistics
            steps.append(i - prev_i)
            rewards.append(epoch_reward)
            prev_i = i
            epoch_reward = 0

    env.close()
    print(f"Played {len(steps)} times.")
    print("Mean steps:{}".format(np.mean(steps)))
    print("Mean reward:{}".format(np.mean(rewards)))