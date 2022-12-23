from .common_imports import *
warnings.filterwarnings('ignore', message='.*ShapelyDeprecationWarning.*')
warnings.simplefilter(action='ignore', category=FutureWarning)
CRS = {
    'census'  : 'EPSG:4269'  , # degrees - used by Census
    'bigquery': 'EPSG:4326'  , # WSG84 - used by Bigquery
    'area'    : 'ESRI:102003', # meters
    'length'  : 'ESRI:102005', # meters
}

@dataclasses.dataclass
class Github():
    repo : str = 'config'
    root : str = '/content/'
    user : str = 'drscook'
    email: str = 'scook@tarleton.edu'
    token: str = ''

    def __post_init__(self):
        os.system(f'git config --global user.email {self.email}')
        os.system(f'git config --global user.name {self.user}')
        if self.token:
            self.url = f'https://{self.token}@github.com/{self.user}/{self.repo}'
        else:
            self.url = f'https://github.com/{self.user}/{self.repo}.git'
        self.path = self.root / self.repo

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
        
def listify(X):
    """Turns almost anything into a list"""
    if isinstance(X, list):
        return X
    elif (X is None) or (X is np.nan) or (X==''):
        return []
    elif isinstance(X, str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]

def cartesian(D):
    D = {key: listify(val) for key, val in D.items()}
    return [dict(zip(D.keys(), x)) for x in it.product(*D.values())]

def mkdir(path, overwrite=False):
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=True, parents=True)

def jsonify(file, obj=None):
    fn = str(file).split('.')[0] + '.json'
    if obj:
        with open(fn, 'w') as outfile:
            json.dump(obj, outfile, indent=4)
    else:
        with open(fn, 'r') as infile:
            return json.load(infile)

def rjust(msg, width, fillchar='0'):
    return str(msg).rjust(width, fillchar)

def ljust(msg, width, fillchar='0'):
    return str(msg).ljust(width, fillchar)

def replace(msg, repls):
    for pats, repl in repls.items():
        for pat in listify(pats):
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

def to_numeric(ser):
    """converts columns to small numeric dtypes when possible"""
    dt = str(ser.dtype)
    if  dt in ['geometry', 'timestamp'] or dt[0].isupper():
        return ser
    else:
        return pd.to_numeric(ser, errors='ignore', downcast='integer')  # cast to numeric datatypes where possible

def prep(df, fix_names=True):
    """prep dataframes"""
    idx = len(df.index.names)
    df = df.reset_index()
    if fix_names:
        df.columns = [c.strip().lower() for c in df.columns]
    return df.apply(to_numeric).convert_dtypes().set_index(df.columns[:idx].tolist())

def transform_labeled(trans, df):
    """apply scikit-learn tranformation and return dataframe with appropriate column names and index"""
    return prep(pd.DataFrame(trans.fit_transform(df), columns=trans.get_feature_names_out(), index=df.index))

def decade(year):
    return int(year) // 10 * 10

class BigQuery():
    def __init__(self, project_id):
        from google.colab import auth
        from google.cloud.bigquery import Client
        auth.authenticate_user()
        self.client = Client(project=project_id)
  
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
              geo = gpd.GeoSeries.from_wkt(df['geometry'], crs=CRS['bigquery'])
              df = gpd.GeoDataFrame(df, geometry=geo)
        return df

    def qry_to_tbl(self, qry, tbl, overwrite=True):
        if not self.get_tbl(tbl, overwrite=overwrite):
            qry = f"""
create table {tbl} as (
    {subquery(qry)}
)"""
            self.client.create_dataset(tbl.split('.')[0], exists_ok=True)
            self.run_qry(qry)
        return tbl

    def df_to_tbl(self, df, tbl, overwrite=True):
        X = df.reset_index().drop(columns=['index', 'level_0'], errors='ignore')
        self.client.create_dataset(tbl.split('.')[0], exists_ok=True)
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
