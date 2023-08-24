import os

lib_path = os.path.abspath(os.path.dirname(__file__))
default_logger_name_ = "huggingbutt-pyci"
cache_path = "~/.cache/huggingbutt"
real_cache_path = os.path.expanduser(cache_path)
zip_path = os.path.join(real_cache_path, "zip")
env_path = os.path.join(real_cache_path, "envs")
agent_path = os.path.join(real_cache_path, "agents")
downloaded_path = os.path.join(zip_path, ".downloaded")
downloaded_env_path = os.path.join(downloaded_path, "env")
downloaded_agent_path = os.path.join(downloaded_path, "agent")
hub_url = "https://huggingbutt.com"
# hub_url = 'http://127.0.0.1:8000'

# todo...
# required parameters in the configuration file
env_config_required = [
    'app.exe_file'
]