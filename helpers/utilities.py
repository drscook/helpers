from .common_imports import *

################################################################################
### Convennient Helper Functions ###
################################################################################

def to_numeric(ser):
    dt = str(ser.dtype).lower()
    if not ('datetime' in dt or 'geometry' in dt or 'timestamp' in dt):
        if 'object' in dt:
            ser = ser.astype('string')
        for _ in range(3):
            # turn strings to numeric if possible & convert to new nullable dtypes
            # repeat because further downcasting might be possible after conversion due to integer type nulls
            ser = pd.to_numeric(ser, errors='ignore', downcast='integer').convert_dtypes()  # cast to numeric datatypes where possible
    return ser.convert_dtypes()

def prep(self, cap='casefold', squeeze=True):
    df = pd.DataFrame(self)
    k = len(df.index.names)
    df = df.reset_index().apply(to_numeric)
    f = {'casefold':lambda s:s.casefold(), 'lower':lambda s:s.lower(), 'upper':lambda s:s.upper(), 'capitalize':lambda s:s.capitalize(), 'title':lambda s:s.title(), None:lambda s:s}
    df.columns = [None if x == 'index' else f[cap](x).strip() for x in df.columns]
    df = df.set_index(df.columns[:k].tolist())
    return df.squeeze() if squeeze else df
pd.DataFrame.prep = prep
pd.Series.prep = prep

def html(self, color='red_dark', odd_bg_color='dark grey', padding='2px', text_align='center', file=None, show=True, **kwargs):
    df = pd.DataFrame(self)
    self.table = pretty_html_table.build_table(df, color=color, odd_bg_color=odd_bg_color, padding=padding, text_align=text_align, **kwargs)
    try:
        with open(file, 'w') as f:
            f.write(self.table)
        print(f'HTML table written to {file}')
    except:
        pass
    if show:
        display(IPython.display.HTML(self.table))
    return self.table
pd.DataFrame.html = html
pd.Series.html = html

def listify(X):
    """Turns almost anything into a list"""
    if X is None or X is np.nan:
        return []
    elif isinstance(X, (list, tuple, set, type({}.keys()), type({}.values()), pd.Index)):
        return list(X)
    elif isinstance(X, dict):
        return [list(X.keys()), list(X.values())]
    elif isinstance(X, np.ndarray):
        return X.tolist()
    elif isinstance(X, pd.DataFrame):
        return X.values.tolist()
    elif isinstance(X, pd.Series):
        return X.tolist()
    else:
        return [X]

def setify(X):
    return set(listify(X))

def cartesian(dct):
    """Creates the Cartesian product of a dictionary with list-like values"""
    D = {key: listify(val) for key, val in dct.items()}
    return [dict(zip(D.keys(), x)) for x in it.product(*D.values())]

def mkdir(path, overwrite=False, exist_ok=True, parents=True):
    """Make dir, overwriting existing if desired"""
    path = pathlib.Path(path)
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=exist_ok, parents=parents)

def jsonify(file, dct=None):
    """Writes dct to file if dct is not None; else, reads file"""
    fn = pathlib.Path(file).with_suffix('.json')
    if dct:
        with open(fn, 'w') as outfile:
            json.dump(dct, outfile, indent=4)
    else:
        with open(fn, 'r') as infile:
            return json.load(infile)

def findn(msg, pat, n=1):
    k = msg.find(pat)
    while n > 1:
        k = msg.find(pat, k+1)
        n -= 1
    return k

def rjust(msg, width, fillchar='0'):
    """Right justifies strings.  Can apply to pandas Series"""
    if isinstance(msg, pd.Series):
        return msg.astype(str).str.rjust(width, fillchar)
    else:
        return str(msg).rjust(width, fillchar)

def ljust(msg, width, fillchar='0'):
    """Left justifies strings.  Can apply to pandas Series"""
    if isinstance(msg, pd.Series):
        return msg.astype(str).str.ljust(width, fillchar)
    else:
        return str(msg).ljust(width, fillchar)

def replace(msg, repls):
    """
    Iterative string replacement on msg, which can either be a string or panda Series of strings
    repl is a dictionary with where each key can be a single pattern or tuple of pattens to be replaced with the value
    """
    for pats, repl in repls.items():
        for pat in listify(pats):
            try:
                # If msg is a pandas series
                msg = msg.str.replace(pat, repl)
            except AttributeError:
                # If msg is a single string
                msg = msg.replace(pat, repl)
    return msg

def join(parts, sep=', '):
    """ join list into single string """
    return sep.join([str(msg) for msg in listify(parts)])

def subquery(qry, indents=1):
    """indent query for inclusion as subquery"""
    s = '\n' + indents * '    '
    return qry.strip(';\n ').replace('\n', s)  # strip leading/trailing whitespace and indent all but first line

def select(cols, indents=1, sep=',\n', tbl=None):
    """ useful for select statements """
    cols = listify(cols)
    if tbl is not None:
        cols = [f'{tbl}.{x}' for x in cols]
    qry = join(cols, sep)
    return subquery(qry, indents)

def transform_labeled(trans, df):
    """apply scikit-learn tranformation and return dataframe with appropriate column names and index"""
    return pd.DataFrame(trans.fit_transform(df), columns=trans.get_feature_names_out(), index=df.index).prep()

def decade(year):
    return int(year) // 10 * 10

def unzipper(file):
    os.system(f'unzip -u -qq -n {file} -d {file.parent}')

def mount_drive(path='/content/drive'):
    import google.colab
    path = pathlib.Path(path)
    google.colab.drive.mount(str(path))
    return path / 'MyDrive'


################################################################################
### Interact With BigQuery ###
################################################################################
@dataclasses.dataclass
class BigQuery():
    """
    ds = dataset
    tbl = table
    qry = query
    """
    project_id: str = 'tarletondatascience2022'
    def __post_init__(self):
        from google.colab import auth
        from google.cloud.bigquery import Client
        auth.authenticate_user()
        self.client = Client(project=self.project_id)
  
    def extract_ds(self, tbl):
        return join(tbl.split('.')[:-1], '.')
    
    def del_tbl(self, tbl):
        self.client.delete_table(tbl, not_found_ok=True)

    def del_ds(self, ds):
        self.client.delete_dataset(ds, not_found_ok=True, delete_contents=True)

    def get_tbl(self, tbl, overwrite=False):
        if overwrite:
            self.del_tbl(tbl)
        try:
            return self.client.get_table(tbl)
        except:
            return False

    def get_schema(self, tbl, overwrite=False):
        t = self.get_tbl(tbl, overwrite)
        if t:
            return t.schema
        else:
            return t
    
    def get_cols(self, tbl, overwrite=False):
        t = self.get_schema(tbl, overwrite)
        if t:
            return [s.name.lower() for s in t]
        else:
            return t

    def copy_tbl(self, curr, targ=None, overwrite=False):
        if targ is None or targ == curr:
            targ = curr + '2'
        if self.get_tbl(targ, overwrite):
            print(f'{targ} exists - use overwrite=True to replace')
        else:
            self.client.copy_table(curr, targ)
            
    def get_ds(self, ds, overwrite=False):
        if overwrite:
            self.del_ds(ds)
        try:
            return self.client.get_dataset(ds)
        except:
            return False

    def copy_ds(self, curr, targ=None, overwrite=False):
        if targ is None or targ == curr:
            targ = curr + '2'
        if self.get_ds(targ, overwrite):
            print(f'{targ} exists - use overwrite=True to replace')
        else:
            self.client.create_dataset(targ)
            for t in self.client.list_tables(curr):
                self.copy_tbl(curr=t, targ=f'{targ}.{t.table_id}')

    def run_qry(self, qry):
        return self.client.query(qry).result()

    def qry_to_df(self, qry):
        res = self.run_qry(qry)
        if res.total_rows > 0:
          res = res.to_dataframe().prep()
          if 'geometry' in res.columns:
                import geopandas as gpd
                geo = gpd.GeoSeries.from_wkt(res['geometry'].astype(str), crs=CRS['bigquery'])
                res = gpd.GeoDataFrame(res, geometry=geo)
        return res

    def qry_to_tbl(self, qry, tbl, overwrite=True):
        if not self.get_tbl(tbl, overwrite=overwrite):
            qry = f"""
create table {tbl} as (
    {subquery(qry)}
)"""
            self.client.create_dataset(self.extract_ds(tbl), exists_ok=True)
            self.run_qry(qry)
        return tbl

    def df_to_tbl(self, df, tbl, overwrite=True):
        X = df.reset_index().drop(columns=['index', 'level_0'], errors='ignore')
        self.client.create_dataset(self.extract_ds(tbl), exists_ok=True)
        t = self.get_schema(tbl, overwrite=overwrite)
        if t:
            self.client.insert_rows_from_dataframe(tbl, X, t)
        else:
            self.client.load_table_from_dataframe(X, tbl).result()
        return tbl

    def tbl_to_df(self, tbl, rows=3):
        try:
            assert rows > 0
            qry = f'select * from {tbl} limit {rows}'
        except:
            qry = f'select * from {tbl}'
        return self.qry_to_df(qry)
    
    def union(self, src, targ, delete=True, distinct=False):
        sep = '\nunion distinct\n' if distinct else '\nunion all\n'
        src = listify(src)
        qry = join(['select * from ' + t for t in src], sep)
        self.qry_to_tbl(qry, targ)
        if delete:
            [self.del_tbl(t) for t in src]

################################################################################
### Interact With Github Repos ###
################################################################################
@dataclasses.dataclass
class Github():
    url  : str
    root_path : str
    user : str = 'drscook'
    email: str = 'scook@tarleton.edu'
    token: str = ''

    def __post_init__(self):
        self.owner, self.name = self.url.split('/')
        os.system(f'git config --global user.email {self.email}')
        os.system(f'git config --global user.name {self.user}')
        if self.token:
            # read & write access
            self.url = f'https://{self.token}@github.com/{self.url}'
        else:
            # read-only access
            self.url = f'https://github.com/{self.url}.git'
        self.path = pathlib.Path(self.root_path) / self.name

    def sync(self, msg='changes'):
        cwd = os.getcwd()
        mkdir(self.path.parent)
        os.chdir(self.path.parent)
        if os.system(f'git clone {self.url}') != 0:
            os.chdir(self.path)
            os.system(f'git remote set-url origin {self.url}')
            os.system(f'git pull')
            os.system(f'git add .')
            os.system(f'git commit -m {msg}')
            os.system(f'git push')
            res = os.popen(f'git commit -m {msg}').read()
            print(res)
            if 'Your branch is ahead of' in res:
                print('you might not have push priveleges to this repo')
        os.chdir(cwd)

def clone_repo(url, path, gitcreds_file='gitcreds.json'):
    gitcreds_file = pathlib.Path(path) / gitcreds_file
    try:
        gitcreds = jsonify(gitcreds_file)
        repo = Github(url, path, **gitcreds)
    except:
        print(f'{gitcreds_file} missing or invalid - using default Github credentials')
        repo = Github(url, path)
    repo.sync()
    sys.path.insert(0, f'{path}/{repo.name}')
    return repo
