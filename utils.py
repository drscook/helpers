from . import *

@dataclasses.dataclass
class Github():
    token: str
    repo : str
    user : str = 'drscook'
    email: str = 'scook@tarleton.edu'
    base : str = '/content/'

    def __post_init__(self):
        self.url = f'https://{self.token}@github.com/{self.user}/{self.repo}'
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


class BQ():
  def __init__(self, project_id):
    from google.colab import auth
    from google.cloud.bigquery import Client
    auth.authenticate_user()
    self.client = Client(project=project_id)
  
  def run_query(self, qry):
      res = self.client.query(qry).result()
      if res.total_rows > 0:
          res = prep(res.to_dataframe())
          if 'geometry' in res.columns:
              geo = gpd.GeoSeries.from_wkt(res['geometry'], crs=CRS['bigquery'])
              res = gpd.GeoDataFrame(res, geometry=geo)
      return res
