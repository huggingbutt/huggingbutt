# HuggingButt Python Client 
Client library to download and publish environments, agents on the huggingbutt.com hub

## Installation
After testing, this package can run stably under python version 3.9 on Windows/MacOS platform. Other versions of python may not be able to install this package,  I will address these issues later on. 

Create a new python environment using anaconda/miniconda.
```shell
conda create -n hb python==3.9
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
git clone xxx
cd huggingbutt
python -m pip install .
```

If there is no error message printed during the installation, congratulations, you have successfully installed this package. Next, you need to apply an access token from the official website http://huggingbutt.com.

Register an account and login, just do as shown in the image below.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/tokens_link.png)

Click new token button generate a new token. This access token is mainly used to restrict the download times of each user, as the server cost is relatively high.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/new_tokens_buttong.png)

Congratulations, you now have an access token!

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/copy_your_token.png)

Just put the generated token in the task code and you're gooooood to go.

Here is a simple training code:
```python
from huggingbutt import Env, set_access_token
from stable_baselines3.ppo import PPO
# your generated access token
ACCESS_TOKEN="YOUR_TOKEN"

if __name__ == '__main__':
    set_access_token(ACCESS_TOKEN)
    env = Env.get('huggingbutt/juggle', 'mac', startup_args=['--time_scale', '10'])
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("ppo_juggle.model")
    env.close()
```
Inference:
```python
from huggingbutt import Agent, Env, set_access_token
# your generated access token
ACCESS_TOKEN="YOUR_TOKEN"

if __name__ == '__main__':
    set_access_token(ACCESS_TOKEN)
    env = Env.get('huggingbutt/juggle', 'mac', startup_args=['--time_scale', '1'])
    agent = Agent.get(20, env)
    obs = env.reset()
    for i in range(1000):
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