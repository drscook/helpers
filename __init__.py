pkgs = (
    'os', 'sys', 'pathlib', 'warnings', 'time', 'json', 'dataclasses', 
    ('itertools', 'it'), ('numpy', 'np'), ('pandas', 'pd'), ('geopandas', 'gpd'),
    ('matplotlib.pyplot', 'plt'),
)
for p in pkgs:
    if isinstance(p, str):
        name, alias = p, p
    elif len(p) > 1:
        name, alias = p[0], p[1]
    else:
        name, alias = p[0], p[0]
    cmd = f'import {name} as {alias}'
    try:
        exec(cmd)
    except ModuleNotFoundError:
        pip = f'pip install {name}'
        print(pip)
        os.system(pip)
        exec(cmd)