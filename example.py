from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('9ba74ddf19b589c0e62a5e773a30a0f2922e3c55')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '10'])

    agent = Agent(
        env=env,
        algorithm='PPO',
        policy='MlpPolicy',
        batch_size=256
    )
    agent.learn(total_timesteps=10000)
    agent.save()
    env.close()

    # agent = Agent.get(20, env)
    #
    # obs = env.reset()
    # for i in range(1000):
    #     act, _status_ = agent.predict(obs)
    #     obs, reward, done, info = env.step(act)
    #     if done:
    #         obs = env.reset()
    # env.close()