import .utils
from google.colab import auth
from google.cloud.bigquery import Client

class BQ():
  def __init__(self, project_id):
    auth.authenticate_user()
    self.client = Client(project=project_id)
  
  def run_query(self, qry):
      res = self.client.query(qry).result()
      if res.total_rows > 0:
          res = utils.prep(res.to_dataframe())
          if 'geometry' in res.columns:
              geo = gpd.GeoSeries.from_wkt(res['geometry'], crs=CRS['bigquery'])
              res = gpd.GeoDataFrame(res, geometry=geo)
      return res
