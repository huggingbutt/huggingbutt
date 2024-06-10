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

from source code
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

Here is a simple inference code:
```python
import numpy as np
from huggingbutt import set_access_token, load_env, load_agent

HB_TOKEN = "YOUR_TOKEN"
set_access_token(HB_TOKEN)

if __name__ == '__main__':
    env = load_env("huggingbutt/roller_ball", "mac", silent=False, num=1, time_scale=20)
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
    env = load_env("huggingbutt/roller_ball", "mac", silent=True, num=3, time_scale=20)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=512,
        tensorboard_log="./logs")
    
    model.learn(total_timesteps=1_000_000)
    model.save(f"roller_ball.zip")
    
    env.close()
```

**I'm incredibly excited for you to upload the environment/agent you trained to huggingbutt.com. It would be amazing for everyone to study!**

## Upload your environment
### 1. Create a workspace directory.
Example:
~/huggingbutt/your_environment_name


### 2. Build your environment
Build your environment created based on tiptoestep. Build the environment in two ways:
1. Regular build mode, which renders the graphical interface. This mode is suitable for demonstrations or use on Windows platform.
![image](https://github.com/huggingbutt/media_store/blob/main/huggingbutt_readme/Screenshot%202024-06-10%20at%2014.05.56.png?raw=true)
2. Dedicated Server build mode, which does not render the graphical interface and is optimized for training speed. However, this mode is only supported on Linux and macOS platforms.Consider naming the environment built under Dedicated Server mode as "your_environment_name_silent"
![image](https://github.com/huggingbutt/media_store/blob/main/huggingbutt_readme/Screenshot%202024-06-10%20at%2014.06.10.png?raw=true)

Filesystem structure:

Mac:
```
└── huggingbutt
    ├── config.toml
    ├── default_functions.py
    ├── your_environment_name.app
    └── your_environment_name_silent
```


###  3. Create config.toml file
Create a config.toml file in "your_environment_name" directory to specify where to load the executable from for the environment. 

Add the following content to config.toml

Mac:
```toml
[app]
	exe_file = "you_environment_name.app/Contents/MacOS/you_environment"
	silent_file = "you_environment_name_silent/you_environment"
	system = "macos"
	engine = "unity"
[action]
	type = "ContinuousAction"
	space = "box"
	shape = 2
	low = -1.0
	high = 1.0
[observation]
	space = "box"
	shape = 12
[function]
	file = "default_functions.py
```

Example:

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/files_list_mac.png)

### 4. Package and compress
You must ensure that after extracting the zip file, config.toml file is located in the root directory.
![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/compress_all_files.png)

### 5. Upload zip file
Click "Add Env"

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/add_env_web.png)

Fill in basic information

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/fill_env_base_info.png)

We will redirect into add new version page after clicking "Save" button.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/new_version.png)

Click "My Envs" will list your environments.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/click_my_env_list.png)


![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/my_env_list.png)


Click the name of your environment will enter the detail page.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/env_detail.png)

Click the blue badge named "* Versions" will list all versions of this environment.

![image](https://raw.githubusercontent.com/huggingbutt/media_store/main/huggingbutt_readme/version_list.png)

Click "New Version" will enter add new version page.

# todo
1. Support more types learning environment, such as native game wrapped by python, pygame, class gym...