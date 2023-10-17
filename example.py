from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('YOUR_TOKEN')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '1'])

    # agent = Agent(
    #     env=env,
    #     algorithm='PPO',
    #     policy='MlpPolicy',
    #     batch_size=64
    # )
    # agent.learn(total_timesteps=20000)
    # agent.save()
    # env.close()

    agent = Agent.get(env, agent_id=15)
    obs = env.reset()
    for i in range(100):
        act, _status_ = agent.predict(obs)
        obs, reward, done, info = env.step(act)
        if done:
            obs = env.reset()
    env.close()
