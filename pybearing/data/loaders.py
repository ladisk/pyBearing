import pandas as pd

from .utils import check_columns


def load_bearing_database_csv(path:str) -> pd.DataFrame:
    """
    Load a bearing database from a CSV file.

    parameters
    ----------
    path : str
        Path to the CSV file containing the bearing database.
    
    returns
    -------
    pd.DataFrame
        DataFrame containing the bearing database.
    """
    df = pd.read_csv(path, sep=',', decimal='.')
    check_columns(df)
    return df
