from importlib.abc import MetaPathFinder
from importlib import util
import subprocess
import sys


class PipFinder(MetaPathFinder):

    def find_spec(self, fullname, path, target=None):
        cmd = f"{sys.executable} -m pip install {self}"
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            return None

        return util.find_spec(self)

    
    
def listify(x):
    return [x]


def draw_histogram(data):
    plt.hist(data)

if __name__ == "__main__":
    sys.meta_path.append(PipFinder)
