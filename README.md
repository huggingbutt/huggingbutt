# HuggingButt
Client library to download and publish reinforcement learning environments/agents on the huggingbutt.com hub.
Join our Discord: https://discord.gg/fnrKcgPR9r

## Installation
The base dependency mlagents-envs has been update to version 1.0.0. However, this version may rise an error [Issue 5996](https://github.com/Unity-Technologies/ml-agents/issues/5996)(on Windows 10). So I specified mlagents-envs==0.30.0, NumPy==1.21.2 in this project. I did not found this issue on MacOS.

Create a new python environment using anaconda/miniconda. Only Python 3.9!!
```shell
conda create -n hb python==3.9.18
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
from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('YOUR_TOKEN')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '1'])
    agent = Agent.get(env, agent_id=15)

    obs = env.reset()
    for i in range(100):
        act, _status_ = agent.predict(obs)
        obs, reward, done, info = env.step(act)
        if done:
            obs = env.reset()
    env.close()
```

Training your agent for this environment. 
```python
from huggingbutt import Env, Agent, set_access_token

if __name__ == '__main__':
    set_access_token('YOUR_TOKEN')
    env = Env.get("huggingbutt/juggle", 'mac', startup_args=['--time_scale', '10'])

    agent = Agent(
        env=env,
        algorithm='PPO',
        policy='MlpPolicy',
        batch_size=64
    )
    agent.learn(total_timesteps=10000)
    agent.save()
    env.close()
```

**I'm incredibly excited for you to upload the environment/agent you trained to huggingbutt.com. It would be amazing for everyone to study!**

## Upload your environment
### 1. Create a workspace directory.
Example:
~/huggingbutt/your_environment_name


### 2. Build your environment
Build your environment created based on Unity ML-Agents, save it to "your_environment_name/unity_package_name"

Filesystem structure:

Mac:
```
└── huggingbutt
    └── your_environment_name
        ├── your_environment_name.app
        └── your_environment_name_BurstDebugInformation_DoNotShip
```

Windows:
```
└── huggingbutt
    └── your_environment_name
        ├── your_environment_name.exe
        └── ...
        └── ...
```


###  3. Create config.toml file
Create a config.toml file in "your_environment_name" directory to specify where to load the executable from for the environment. 

Filesystem structure:
```
└── huggingbutt
    ├── config.toml
    └── your_environment_name
        ├── your_environment_name.app
        └── your_environment_name_BurstDebugInformation_DoNotShip
```

Add the following content to config.toml

Mac:
```toml
[app]
  exe_file="unity_package_name/game.app"
```
Windows:
```toml
[app]
  exe_file="unity_package_name/game.exe"
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
2. Develop a framework, user can customize the observation, action and reward of the environment developed under this framework. I hope this makes it easier for everyone to iterate the environment's agent.
3. There are still many ideas, I will add them later when I think about them...