"""
QREM IO Module
==============

The `qrem.common.io` module provides utility functions for handling dates and file operations, 
specifically for saving and loading data in a QREM (Quantum Error Mitigation) context. It includes 
functions to format current date and time, prepare output file paths, and perform file operations 
like saving and loading data using pickle serialization.

Functions
---------
date_time_formatted()
    Returns the current date and time formatted for file naming.

date_formatted()
    Returns the current date formatted.

date_time_fileformatted()
    Returns the current date and time formatted for file naming, with a file-safe format.

prepare_outfile(outpath: Union[bool, str, Path], overwrite: bool, default_filename: str)
    Prepares and returns a path for saving output files, with optional overwriting.

load(file_path: str)
    Loads data from a pickled file.

save(dictionary_to_save: dict, directory: Union[Path, str], custom_filename: Optional[str], 
     overwrite: bool, verbose_log: bool)
    Saves a dictionary to a pickled file in a specified directory.

    
    
Notes
-----
    @authors: Jan Tuziemski, Filip Maciejewski, Joanna Majsak, Oskar Słowik, Marcin Kotowski, Katarzyna Kowalczyk-Murynka, Paweł Przewłocki, Piotr Podziemski, Michał Oszmaniec
    @contact: michal.oszmaniec@cft.edu.pl
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
    """
    Returns the current date and time formatted as a string suitable for use in filenames.

    Returns
    -------
    str
        The current date and time in "YYYY_MM_DD-HH_MM_SS" format.
    """    
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    
def date_formatted():
    """
    Returns the current date formatted as a string.

    Returns
    -------
    str
        The current date in "YYYY-MM-DD" format.
    """    
    return datetime.now().strftime("%Y-%m-%d")

def date_time_fileformatted():
    """
    Returns the current date and time formatted as a string suitable for filenames,
    ensuring file system compatibility.

    Returns
    -------
    str
        The current date and time in "YYYY_MM_DD__HH_MM_SS" format.
    """    
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

def prepare_outfile(outpath: Union[bool,str,Path]=False,
                    overwrite: bool = False,
                    default_filename = "output.pkl"):
    """
    Prepares and returns a path for an output file. If the path doesn't exist, it will be created.

    Parameters
    ----------
    outpath : Union[bool, str, Path], optional
        The desired path for the output file. If False, a default path will be used.
    overwrite : bool, optional
        If True, will overwrite existing files at the path.
    default_filename : str, optional
        The default filename to use if no specific path is provided.

    Returns
    -------
    Path
        The prepared Path object for the output file.
    """

    if isinstance(outpath, str):
        outpath = Path(outpath) 
    if isinstance(outpath, bool):
        outpath = Path.home().joinpath("qrem_results") #makes a dir path
        
    if isinstance(outpath, Path):
        export_path = outpath
        if not export_path.exists():
            if export_path.suffix:
                export_path.parent.mkdir(parents=True, exist_ok=True)  
            else:   
                export_path.mkdir(parents=True, exist_ok=True)
        if export_path.is_dir():
            export_path = export_path.joinpath(default_filename)
            
    else: 
        export_path = Path.home().joinpath("qrem_results").joinpath(default_filename)
        printer.errprint("Wrong outpath type for backing up data")

    if not overwrite and export_path.exists():
        raise FileExistsError(f"File {export_path} already exists. Set overwrite=True to overwrite.")

    return export_path

def load(file_path: str):
    """
    Loads and returns data from a pickled file.

    Parameters
    ----------
    file_path : str
        The path to the pickled file.

    Returns
    -------
    Any
        The data unpickled from the file.
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

def save(dictionary_to_save: dict,
            directory: Union[Path,str],
            custom_filename: Optional[str] = None,
            overwrite: bool = False,
            verbose_log: bool = True):
    """
    Saves a dictionary to a pickled file in the specified directory.

    Parameters
    ----------
    dictionary_to_save : dict
        The dictionary to be saved.
    directory : Union[Path, str]
        The directory where the file should be saved.
    custom_filename : Optional[str], optional
        A custom filename for the saved file. If None, a default name is generated.
    overwrite : bool, optional
        If True, will overwrite existing data in the directory.
    verbose_log : bool, optional
        If True, prints additional information during the save process.

    Notes
    -----
    The function will create the directory if it does not exist.
    """
    
    if isinstance(directory,str):
        directory = Path(directory)

   


    if directory :
        dir = Path(directory) 
        dir.mkdir(mode=0o777, exist_ok=True, parents= True) 
    else:
        dir = Path.home().joinpath("qrem_results")
        dir.mkdir(mode=0o777, exist_ok=True) 

    if verbose_log:
        printer.qprint("Saving pickled file to: ", str(dir))
    
    if verbose_log:
        printer.qprint("Saving pickled file to: ", str(dir))
    
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


#===========================
# TESTS
#===========================

def _t1():
    path = prepare_outfile(True,True)
    print(path)

if __name__ == "__main__":
    _t1()
