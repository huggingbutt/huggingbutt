# HuggingButt
Client library to download and publish reinforcement learning environments/agents on the huggingbutt.com hub.
Join our Discord: https://discord.gg/fnrKcgPR9r

## Installation
Removed dependency on ml-agents and added support for tiptoestep in the current version.

Create a new python environment using anaconda/miniconda. 
```shell
conda create -n hb
```

activate the new python environment.
```shell
conda activate hb
```

install huggingbutt from pypi
```shell
pip install huggingbutt==0.0.5
```
or from source code
```shell
git clone -b v0.0.5 https://github.com/huggingbutt/huggingbutt.git
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

Here is a simple inference code:
```python
import numpy as np
from huggingbutt import set_access_token, load_env, load_agent

HB_TOKEN = "YOUR_TOKEN"
set_access_token(HB_TOKEN)

if __name__ == '__main__':
    env = load_env("huggingbutt/roller_ball", "mac", silent=True, num=1, time_scale=20)
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
```

Training your agent for this environment. 
```python
import numpy as np
from huggingbutt import set_access_token, load_env, load_agent

HB_TOKEN = "YOUR_TOKEN"
set_access_token(HB_TOKEN)

if __name__ == '__main__':
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=1024,
        tensorboard_log="./logs")
    
    model.learn(total_timesteps=1_000_000)
    model.save(f"roller_ball.zip")
    
    env.close()
```

# todo
1. Support more types learning environment, such as native game wrapped by python, pygame, class gym...