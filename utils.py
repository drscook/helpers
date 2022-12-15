from . import *

@dataclasses.dataclass
class Github():
    token: str
    repo : str
    user : str = 'drscook'
    email: str = 'scook@tarleton.edu'
    base : str = '/content/'

    def __post_init__(self):
        self.url = f'https://{self.access}@github.com/{self.user}/{self.repo}'
        self.path = self.base + self.repo
        os.system(f'git config --global user.email {self.email}')
        os.system(f'git config --global user.name {self.user}')

    def pull(self):
        cwd = os.getcwd()
        os.chdir(self.base)
        if os.system(f'git clone {self.url}') != 0:
            os.chdir(self.path)
            os.system(f'git pull')
        os.chdir(cwd)

    def push(self, msg='changes'):
        cwd = os.getcwd()
        os.chdir(self.path)
        os.system(f'git add .')
        os.system(f'git commit -m {msg}')
        os.system(f'git push')
        os.chdir(cwd)
