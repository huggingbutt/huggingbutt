# HuggingButt
Client library to download and publish reinforcement learning environments/agents on the huggingbutt.com hub

## Installation
This project only supports Python 3.10, and the highest version is 3.10.12. This is due to restrictions from the dependency 'mlagents-envs'. I'll address this issue in subsequent release.

Create a new python environment using anaconda/miniconda.
```shell
conda create -n hb python==3.10.12
```

activate the new python environment.
```shell
conda activate hb
```

install huggingbutt from pypi
```shell
pip install huggingbutt
```
or from source code
```shell
git clone https://github.com/huggingbutt/huggingbutt.git
cd huggingbutt
python -m pip install .
```

If there is no error message printed during the installation, congratulations, you have successfully installed this package. Next, you need to apply an access token from the official website https://huggingbutt.com.

Register an account and login, just do as shown in the image below.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/tokens_link.png)

Click new token button generate a new token. This access token is mainly used to restrict the download times of each user, as the server cost is relatively high.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/new_tokens_buttong.png)

Congratulations, you now have an access token!

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/copy_your_token.png)

Just put the generated token in the task code and you're gooooood to go.

Here is a simple training code:
```python
from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('YOUR_TOKEN')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '1'])

    agent = Agent(
        env=env,
        algorithm='PPO',
        policy='MlpPolicy',
        batch_size=256
    )
    agent.learn(total_timesteps=10000)
    agent.save()
    env.close()
```
Inference:
```python
from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('YOUR_TOKEN')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '1'])
    agent = Agent.get(20, env)

    obs = env.reset()
    for i in range(100):
        act, _status_ = agent.predict(obs)
        obs, reward, done, info = env.step(act)
        if done:
            obs = env.reset()
    env.close()
```

# todo
1. Support more types learning environment, such as native game wrapped by python, pygame, class gym...
2. Develop a framework, user can customize the observation, action and reward of the environment developed under this framework. We hope this makes it easier for everyone to iterate the environment's agent.
3. There are still many ideas, I will add them later when I think about them...