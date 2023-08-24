class HBException(Exception):
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return self.message

class EnvNameErrorException(Exception):
    def __init__(self):
        self.message = "Use name or Env name parse error."

    def __str__(self):
        return self.message

class AccessTokenNotFoundException(Exception):
    """
    Raised when the access token does not found.
    """
    def __init__(self, message=""):
        if not message:
            self.message = "Access token does not found."

    def __str__(self):
        return self.message


class VersionNotFoundException(Exception):
    def __init__(self, message=""):
        if not message:
            self.message = "An error occurred while fetching the latest version remotely."

    def __str__(self):
        return self.message


class HubAccessException(Exception):
    """
    Raised when can not access agenthub service.
    """
    def __init__(self, message="", code=-1):
        if not message:
            message = "Access Error!"
        self.message = message
        self.code = code

    def __str__(self):
        return f"Response Status:{self.code}, Message:{self.message}"


class ConfigParseErrorException(Exception):
    def __init__(self, message="", code=-1):
        if not message:
            message = "Config File Parse Error!"
        self.message = message
        self.code = code

    def __str__(self):
        return f"Configure File Parse Error, {self.message}"
