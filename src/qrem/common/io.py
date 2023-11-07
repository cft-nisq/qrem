"""qrem.common.math module contains helper functions, that can be used to save and load 
intermediate or final results.
"""

from datetime import datetime
from typing import List, Dict, Union, Optional, Callable, Tuple

from pathlib import Path
import shutil
from qrem.common import printer
import pickle
#===========================
# Helpers for dates in path safe mode
#===========================

def date_time_formatted():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
def date_formatted():
    return datetime.now().strftime("%Y-%m-%d")

def date_time_fileformatted():
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")


def prepare_outfile(outpath: Union[bool,str,Path]=False,
                    overwrite: bool = False,
                    default_filename = "output.pkl"):
    

    if isinstance(outpath, str):
        backup = Path(outpath) 
    if isinstance(outpath, bool):
        backup = Path.home().joinpath("qrem_results") #makes a dir path
        
    if isinstance(outpath, Path):
        export_path = outpath
        if not export_path.exists() or export_path.is_dir():
            export_path.mkdir(parents=True, exist_ok=True)
            export_path.joinpath(default_filename)
        if export_path.is_file() :
            export_path.parent.mkdir(parents=True, exist_ok=True)  
    else: 
        export_path =Path.home().joinpath("qrem_results").joinpath(default_filename)
        printer.errprint("Wrong outpath type for backing up data")

    if not overwrite:
        assert (not export_path.exists())

    return export_path
#(PP) delete not used - maybe better than above?
def load(file_path: str):
    """
    Quickly load data from a pickled file containing a dictionary.
    """
    import pickle

    if file_path[-4:] != '.pkl':
        file_path = file_path + '.pkl'
    try:
        with open(file_path, 'rb') as f:
            unpickled = pickle.load(f)
    except(ValueError):
        with open(file_path, 'rb') as f:
            unpickled = pickle.load(f)

    return unpickled





#MOVE_TO >> core.io TO DELETE PP CAN BE A MARKER FOR UNUSED STUFF
def save(dictionary_to_save: dict,
            directory: Union[Path,str],
            custom_filename: Optional[str] = None,
            overwrite: bool = False,
            verbose_log: bool = True):
    """
    Quickly save data to a pickled file containing a dictionary.
    """
    
    if isinstance(directory,str):
        directory = Path(directory)

    if verbose_log:
        printer.qprint("Saving pickled file to: ", str(dir))


    if directory :
        dir = Path(directory) 
        dir.mkdir(mode=0o777, exist_ok=True, parents= True) 
    else:
        dir = Path.home().joinpath("qrem_results")
        dir.mkdir(mode=0o777, exist_ok=True) 
    
    # original_umask = os.umask(0)
    # os.umask(original_umask)


    if dir.is_dir():
        if any(dir.iterdir()) and overwrite:
            printer.warprint("Warning: overwriting folder", f"content of the folder {dir} has been overwritten")
            shutil.rmtree(dir)
            dir.mkdir(mode=0o777, exist_ok=False, parents= True) 
        elif not any(dir.iterdir()) and overwrite:
            printer.errprint("Error: folder exists", f"content of the folder {dir} has been overwritten")
            return
        

    if custom_filename is None:
        dt = date_time_fileformatted()
        file_path = dir.joinpath('Results_Object' + '___' + str(dt) + '.pkl')
    else:
        if custom_filename.split(".")[-1] != '.pkl':
            custom_filename = custom_filename + '.pkl'
        file_path = dir.joinpath(custom_filename)


    with open(file_path, 'wb') as f:
        pickle.dump(dictionary_to_save, f, pickle.HIGHEST_PROTOCOL)