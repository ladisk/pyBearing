import pandas as pd
import importlib.resources as pkg_resources

from .utils import REQUIRED_COLUMNS
from ..core.bearing import Bearing
from ..core.fault_frequencies import FaultFrequencies


class BearingDatabase:
    def __init__(self, load_default:bool = True):
        """
        Class for managing bearing databases. By default, it loads a CSV file embedded in this package,
        which contains a small database of common bearings. User can also import their own database.
        
        parameters
        ----------
        load_default : bool
            If True, loads the default database embedded in this package. If False, creates an empty database.
        """
        # Load the CSV embedded in this package
        if load_default:
            with pkg_resources.files(__package__).joinpath("database.csv").open("r", encoding="utf-8") as f:
                self.database = pd.read_csv(f, sep=',', decimal='.')
        else:
            self.database = pd.DataFrame(columns=REQUIRED_COLUMNS)
        
    def find_bearing_by_name(self, name:str) -> list[Bearing]:
        """
        Find a bearing in the database by its name. If multiple bearings have the same name, all of them will be returned.

        parameters
        ----------
        name : str
            The name of the bearing to find.
        
        returns
        -------
        list[Bearing]
            A list of Bearing objects containing the bearing data if found, or an empty list if not found.
        """
        bearing_data = self.database[self.database["name"] == name]
        if bearing_data.empty:
            return []
        
        return [Bearing.from_dict(row) for _, row in bearing_data.iterrows()]
    
    def find_characteristic_frequencies_by_bearing_name(self, name:str) -> list[FaultFrequencies]:
        """
        Find the characteristic frequencies for a bearing in the database by its name.

        parameters
        ----------
        name : str
            The name of the bearing for which to find characteristic frequencies.

        returns
        -------
        list[FaultFrequencies]
            A list of FaultFrequencies objects containing the characteristic frequencies if found, or an empty list if not found.
        """
        bearing_data = self.database[self.database["name"] == name]
        if bearing_data.empty:
            return []
        
        return [FaultFrequencies.from_dict_database(row) for _, row in bearing_data.iterrows()]

    def export_database(self, path:str, file_name:str) -> bool:
        """
        Export the current bearing database to a CSV file.

        parameters
        ----------
        path : str
            The directory where the file will be saved.
        file_name : str
            The name of the file to be saved.
        """
        if file_name.endswith(".csv"):
            file_name = file_name[:-4]  # Remove .csv extension if provided
        if len(file_name.split(".")) != 1:
            raise ValueError("file_name should end with .csv extension or have no extension at all.")
        self.database.to_csv(f"{path}/{file_name}.csv", index=False)
        return True

    def set_database(self, df:pd.DataFrame) -> bool:
        """
        Set a new bearing database from a DataFrame. If function for loading data into DataFrame
        is not form this package, user should check if the DataFrame contains required columns using
        check_columns() function from pybearing.data.

        parameters
        ----------
        df: pd.DataFrame
            DataFrame containing new database.
        """
        self.database = df
        return True
