from huggingbutt import Env, Agent, set_access_token



if __name__ == '__main__':
    set_access_token('08d4db07fe2532a5b629bf7aaf3dc6bc2aba35d3')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '10'])
    agent = Agent(
        algorithm='PPO',
        policy='MlpPolicy',
        env=env
    )

    agent.learn(total_timesteps=1000)

    agent.save()
    env.close()





