from .common_imports import *

def to_numeric(ser):
    """converts columns to small numeric dtypes when possible"""
    dt = str(ser.dtype)
    if  dt in ['geometry', 'timestamp'] or dt[0].isupper():
        return ser
    else:
        return pd.to_numeric(ser, errors='ignore', downcast='integer')  # cast to numeric datatypes where possible

def listify(X):
    """Turns almost anything into a list"""
    if X is None or X is np.nan:
        return []
    elif isinstance(X, (list, tuple, set, pd.Index)):
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

def prep(X, mode='lower'):
    """Common data preparation such as standardizing capitalization"""
    modes = ['lower', 'capitalize', 'casefold', 'swapcase', 'title', 'upper', None, False]
    assert mode in modes, f'mode must one of {modes} ... got {mode}'
    if X is None or X is np.nan:
        return None
    elif isinstance(X, str):
        if mode:
            X = getattr(X, mode)()
        return X.strip()
    elif isinstance(X, (list, tuple, set, pd.Index)):
        return type(X)((prep(x, mode) for x in X))
    elif isinstance(X, dict):
        return dict(zip(*prep(listify(X), mode)))
    elif isinstance(X, np.ndarray):
        return np.array(prep(listify(X), mode))
    elif isinstance(X, pd.DataFrame):
        idx = len(X.index.names)
        X = X.reset_index()
        X.columns = prep(X.columns, mode)
        return X.apply(to_numeric).convert_dtypes().set_index(X.columns[:idx].tolist())
    elif isinstance(X, pd.Series):
        return prep(X.to_frame(), mode).squeeze()
    else:
        return X

def cartesian(dct):
    """Creates the Cartesian product of a dictionary with list-like values"""
    D = {key: listify(val) for key, val in dct.items()}
    return [dict(zip(D.keys(), x)) for x in it.product(*D.values())]

def mkdir(path, overwrite=False):
    """Make dir, overwriting existing if desired"""
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)

def jsonify(file, dct=None):
    """Writes dct to file if dct is not None; else, reads file"""
    fn = pathlib.Path(file).with_suffix('.json')
    if obj:
        with open(fn, 'w') as outfile:
            json.dump(dct, outfile, indent=4)
    else:
        with open(fn, 'r') as infile:
            return json.load(infile)

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

def make_select(cols, indents=1, sep=',\n', tbl=None):
    """ useful for select statements """
    cols = listify(cols)
    if tbl is not None:
        cols = [f'{tbl}.{x}' for x in cols]
    qry = join(cols, sep)
    return subquery(qry, indents)

def transform_labeled(trans, df):
    """apply scikit-learn tranformation and return dataframe with appropriate column names and index"""
    return prep(pd.DataFrame(trans.fit_transform(df), columns=trans.get_feature_names_out(), index=df.index))

def decade(year):
    return int(year) // 10 * 10

@dataclasses.dataclass
class BigQuery():
    project_id: str

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
          df = prep(res.to_dataframe())
          if 'geometry' in df.columns:
                import geopandas as gpd
                geo = gpd.GeoSeries.from_wkt(df['geometry'].astype(str), crs=CRS['bigquery'])
                df = gpd.GeoDataFrame(df, geometry=geo)
        return df

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


@dataclasses.dataclass
class Github():
    repo_url  : str
    root_path : str
    user : str = 'drscook'
    email: str = 'scook@tarleton.edu'
    token: str = ''

    def __post_init__(self):
        os.system(f'git config --global user.email {self.email}')
        os.system(f'git config --global user.name {self.user}')
        if self.token:
            # let's us push changes to repo
            self.url = f'https://{self.token}@github.com/{self.user}/{self.repo_url}'
        else:
            # read-only access
            self.url = f'https://github.com/{self.user}/{self.repo}.git'
            self.url = f'https://github.com/{self.user}/{self.repo}.git'
        self.path = self.root_path / self.repo

    def sync(self, msg='changes'):
        cwd = os.getcwd()
        self.root.mkdir(exist_ok=True, parents=True)
        os.chdir(self.root)
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
