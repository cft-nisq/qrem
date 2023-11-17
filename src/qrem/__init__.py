#ORGANIZATION - dont know what is the purpose of this file (MO) - (PP): important for defining initialization for packages/subpackages in python (think of every folder as a package/subpackage, when you import it __init__.py is run)
"""**qrem** module is the main module of the whole package.
It contains all qrem code after refactoring into upcoming version 1.0.0. Currently it holds
most important data structures and common functions used in the freamework. The main subpackages are: \n\n
- **common**: contains all common functions, utilities, connection to quantum backends etc. used throughout whole qrem package\n
- **cn**: csubpackage containing cluster-neighbours approach implementation \n
- **ctmp**: Mitigation procedure as described in Bravyi et al. \n
- **pipelines**: scripts containing executions of most common characterisation/mitigation routines from data generation, sending to quantum machine down to characterization and mitigation. \n
- **types**: most important data holder classess used throughout the package \n
""" 


from importlib.metadata import version
from dotenv import load_dotenv

from qrem.common.printer import qprint
from qrem.common.config import QremConfigLoader
#-----------------------
# [1] read version from installed package and provide in __version__ variable
#-----------------------
__version__ = version("qrem")

#-----------------------
# [2] read environment variables.
#-----------------------
# Set up environment variables based on the .env file https://dev.to/jakewitcher/using-env-files-for-environment-variables-in-python-applications-55a1
load_dotenv()


#-----------------------
# [3] modules binding for top-level qrem package
#-----------------------
# add here anything that should be accesible directly from "qrem.":
# example: if you want "import qrem.core.something.else" module  be available for users as "import qrem.else"

#-----------------------
# [4] MAIN PACKAGE FUNCTIONS
#-----------------------
#Main package operational  functions
def load_config(cmd_args = None, path:str = None, verbose_log = False):
    """Loads config for running main QREM pipeline

    Parameters
    ----------
    path: str, default = None
        Path to the config file to load
    verbose_log: bool, default=False
        turn on verbose logging for more printouts with info
    """

    if cmd_args != None:
        qprint(f"- LOADING CONFIG VIA CMD ARGS: ", f"{cmd_args}")
        config = QremConfigLoader.load(cmd_args=cmd_args)
    elif path != None:
        qprint(f"- LOADING CONFIG VIA FILE PATH: ", f"{path}")
        config = QremConfigLoader.load(default_path=path)
    else:
        config = QremConfigLoader.load()
    
    if(verbose_log):
        QremConfigLoader.values()

    return config


# #Main package operational  functions
# def load_config(path:str = None, verbose_log = False):
#     """Loads config for running main QREM pipeline

#     Parameters
#     ----------
#     path: str, default = None
#         Path to the config file to load
#     verbose_log: bool, default=False
#         turn on verbose logging for more printouts with info
#     """

#     if path != None:
#         print(path)
#         config = QremConfigLoader.load(default_path=path, as_dict=False)
#     else:
#         config = QremConfigLoader.load()
    
#     if(verbose_log):
#         QremConfigLoader.values()

#     return config