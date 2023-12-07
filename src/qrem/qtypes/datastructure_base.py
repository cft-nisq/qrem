"""
Data Structure Module

This module provides a base class for data structures with export and import capabilities.
It includes methods for exporting and importing data structures to/from JSON and pickle formats.

Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
"""
from pathlib import Path
import pickle

import orjson
import numpy as np

from qrem.common import printer



class DataStructureBase:
    """
    Base class for data structures with export and import capabilities.

    Attributes
    ----------
    None

    Methods
    -------
    get_dict_format()
        Returns this class as a JSON-like dictionary structure.

    load_from_dict(dictionary)
        Loads class fields from a dictionary.

    to_json()
        Returns this class as a JSON structure.

    export_json(json_export_path, overwrite=False)
        Saves the class into a JSON file.

    import_json(json_import_path)
        Imports class fields from a JSON file.

    get_pickle()
        Returns this class as a pickle structure.

    export_pickle(pickle_export_path, overwrite=False)
        Saves the class into a pickle file.

    import_pickle(pickle_import_path)
        Imports class fields from a pickle file.
    """

    def get_dict_format(self):
        """returns this class as a json-like dictionary structure"""
        return self.__dict__

    def load_from_dict(self, dictionary):
        for key in dictionary:
            if key in self.get_dict_format():
                setattr(self, key,
                        dictionary[key])
        pass

    def to_json(self):
        """returns this class as a json structure"""
        return orjson.dumps(self, default=lambda o: o.__dict__,
                            option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SORT_KEYS)

    def export_json(self, json_export_path, overwrite=False):
        """saves the class into json file"""
        if Path(json_export_path).is_file() and not overwrite:
            raise FileExistsError(f"File already exists: <{json_export_path}>")
        elif Path(json_export_path).is_dir():
            raise FileExistsError(f"Path is a directory: <{json_export_path}>, provide a path to a file")
        elif not Path(json_export_path).parent.is_dir():
            Path(json_export_path).parent.mkdir(parents=True, exist_ok=True)

        with open(json_export_path, 'wb') as outfile:
            outfile.write(self.to_json())

    def import_json(self, json_import_path):
        """imports class' fields from json file"""
        with open(json_import_path, 'rb') as outfile:
            json_dict = orjson.loads(outfile.read())
        self.load_from_dict(json_dict)

    def get_pickle(self):
        """returns this class as a pickle structure"""
        return pickle.dumps(self.get_dict_format())


    def export_pickle(self, pickle_export_path, overwrite=False):
        """saves the class into pickle file"""

        
        if Path(pickle_export_path).is_file() and not overwrite:
            raise FileExistsError(f"File already exists: <{pickle_export_path}>")
        elif Path(pickle_export_path).is_dir():
            raise FileExistsError(f"Path is a directory: <{pickle_export_path}>, provide a path to a file")
        elif not Path(pickle_export_path).parent.is_dir():
            Path(pickle_export_path).parent.mkdir(parents=True, exist_ok=True)

        with open(pickle_export_path, 'wb') as outfile:
            pickle.dump(self.get_dict_format(), outfile, pickle.HIGHEST_PROTOCOL)

    def import_pickle(self, pickle_import_path):
        """imports class' fields from pickle file"""
        
        if file_path[-4:] != '.pkl':
            file_path = file_path + '.pkl'
        try:
            with open(pickle_import_path, 'rb') as outfile:
                pickle_dict = pickle.load(outfile)
            self.load_from_dict(pickle_dict)
        except(ValueError):
            with open(file_path, 'rb') as f:
                pickle_dict = pickle.load(outfile)


class MyTestStructure(DataStructureBase):
    """
    Example custom data structure derived from DataStructureBase.

    Attributes
    ----------
    data : numpy.ndarray
        The data stored in this structure.

    Methods
    -------
    __init__(data)
        Initializes a new instance of MyStructure.

    """    
    def __init__(self,data):
        super().__init__()
        self.data = data

if __name__ == "__main__":
    _testcase1 = MyTestStructure(np.random.randint(10,size =10))
    print(_testcase1.to_json())