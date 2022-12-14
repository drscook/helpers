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

def run_query(qry):
    res = client.query(qry).result()
    if res.total_rows > 0:
        res = prep(res.to_dataframe())
        if 'geometry' in res.columns:
            geo = gpd.GeoSeries.from_wkt(res['geometry'], crs=CRS['bigquery'])
            res = gpd.GeoDataFrame(res, geometry=geo)
    return res
