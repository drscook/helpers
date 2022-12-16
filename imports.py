pkgs = (
    'os',
    'sys',
    'pathlib',
    'warnings',
    'time',
    'json',
    'dataclasses', 
    ('itertools', 'it'),
    ('numpy', 'np'),
    ('pandas', 'pd'),
    ('geopandas', 'gpd'),
    ('matplotlib.pyplot', 'plt'),
)
def pip_import(pkg):
    if isinstance(p, str):
        name, alias = pkg, pkg
    elif len(p) > 1:
        name, alias = pkg[0], pkg[1]
    else:
        name, alias = pkg[0], pkg[0]
    cmd = f'import {name} as {alias}'
    try:
        exec(cmd)
    except ModuleNotFoundError:
        print('pip installing ', name)
        os.system(f'pip install {name}')
    return cmd

for pkg in pkgs:
    exec(pip_import(pkg))
