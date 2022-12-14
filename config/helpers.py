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
