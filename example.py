from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('08d4db07fe2532a5b629bf7aaf3dc6bc2aba35d3')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '1'])

    # agent = Agent(
    #     env=env,
    #     algorithm='PPO',
    #     policy='MlpPolicy',
    #     batch_size=256
    # )
    # agent.learn(total_timesteps=200000)
    # agent.save()
    # env.close()
    # agent = Agent.get(19, env)
    # print(agent)
    # env.close()

    agent = Agent.get(20, env)

    obs = env.reset()
    for i in range(1000):
        act, _status_ = agent.predict(obs)
        obs, reward, done, info = env.step(act)
        if done:
            obs = env.reset()

    env.close()