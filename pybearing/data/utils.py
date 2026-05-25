import pandas as pd


REQUIRED_COLUMNS = ["name", "manufacturer", "d", "D", "B", "D_b", "D_m", "theta", "N_b", "f_rot", "f_c", "f_re_ax", "f_o", "f_re", "f_i"]

def check_columns(df:pd.DataFrame) -> bool:
    """
    Check if the database converted in df contains required columns for other functionality of the package.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing new database.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise KeyError(f"Column '{missing_columns}' missing, but required.")
    return True