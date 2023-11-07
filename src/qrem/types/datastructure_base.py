"""class should:

    should work for any field that could appear on a class and work for numpyarrays
    to_json() function that will return dict with class __ dict __ dump
    export_json(path)
    import_json(path)
    export_pickle(path)
    import_pickle(path)
"""
from pathlib import Path
import pickle

import orjson
import numpy as np



class DataStructureBase:
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
            print(f"WARNING:: Omitting export to existing file: <{json_export_path}>")
        else:
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
            print(f"WARNING:: Omitting export to existing file: <{pickle_export_path}>")
        else:
            with open(pickle_export_path, 'wb') as outfile:
                pickle.dump(self.get_dict_format(), outfile)

    def import_pickle(self, pickle_import_path):
        """imports class' fields from pickle file"""
        with open(pickle_import_path, 'rb') as outfile:
            pickle_dict = pickle.load(outfile)
        self.load_from_dict(pickle_dict)

class MyStructure(DataStructureBase):
    def __init__(self,data):
        super().__init__()
        self.data = data

if __name__ == "__main__":
    testcase1 = MyStructure(np.random.randint(10,size =10))
    print(testcase1.to_json())