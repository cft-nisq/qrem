# QREM - Quantum Readout Errors Mitigation

This code accompanies the paper **[Efficient reconstruction, benchmarking and validation of cross-talk models in readout noise in near-term quantum devices](https://arxiv.org/)**. 

The purpose of this git  branch is to keep the exact codebase that was used for development of the article, while the qrem package will continue to be updtated in its main branch [here](https://github.com/cft-nisq/qrem).

For the main README.md file with the description of the package look [here](https://github.com/cft-nisq/qrem#readme)

## Usage

In order to perform analysis the following steps needed to be taken:

1. Clone the repository and set up the project.

    ```bash
    git clone git@github.com:cft-nisq/qrem.git
    ```

    or - of you already have an installation:

    ```bash
    git pull
    ```

2. Make sure that you have Python newer than 3.9.0 and newest version of pip. If not - best to upgrade both before proceeding further.

3. We suggest first to create a separate virtual environment using venv. From qrem main repository  path create a new venv:

    ```bash
    python -m venv venv_qrem

    (on most Linux/MacOS installations of python use python3)

    python3 -m venv venv_qrem
    ```

4. Activate your virtual environment. To do that on Windows, run (still beeing in QREM_SECRET_DEVELOPMENT folder):

    ```bash
    On windows:
    ---------------
    .\venv_qrem\Scripts\activate

    On Linux or MacOS run:
    ---------------
    source venv_qrem/bin/activate
    ```

    If the command does not work, you need to fix execution permission on the file located in  *\venv_qrem\Scripts\activate* or *source venv_qrem/bin/activate*. The method is specific for OS - should be easy to find in google.

5. Now you have two options. You can install QREM package in a developer (editable) mode so it will be easy for you to modify the package code, or you can install qrem traditionally, from pip. We suggest OPTION 1 (the qrem package version on pypi will be updated in the future).

   1. OPTION 1 *Installing qrem in editable/dev mode*

      1. Upgrade pip (if it is not upgraded):

          ```bash
          On windows:
          ---------------
          .\venv_qrem\Scripts\python.exe -m pip  install --upgrade pip

          
          On Linux or MacOS run:
          ---------------
          venv_xxxx/Scripts/python3 pip install --upgrade pip
          ```

      2. Install qrem package in development mode from local:

          ```bash
          On windows:
          ---------------
          pip install --editable .

          On Linux or MacOS run:
          ---------------
          pip3 install --editable .
          ```

   2. OPTION 2 *Installing qrem from pypi*

      ```bash
      On windows:
      ---------------
      pip install qrem

      On Linux or MacOS run:
      ---------------
      pip3 install qrem
      ```

6. Download experimental data available online [here](https://drive.google.com/drive/folders/14Jh3gJUbiipVLVoWSugJ4uYcZpZWd9XS?usp=drive_link)

7. Set paths to experimental data in the article_data_analysis.py:

    * **DATA_DIRECTORY**: directory with experimental data

    * **FILE_NAME_RESULTS_IBM** and **FILE_NAME_RESULTS_RIG**: files with raw experimental results, on Google drive linked above *RESULTS_IBM_CUSCO.pkl* and *RESULTS_RIGETTI_ASPEN-M-3.pkl*

    * **FILE_NAME_GROUND_STATES_LIST_IBM** and **FILE_NAME_GROUND_STATES_LIST_RIG**: files with ground states used in benchmarks, on Google drive linked above  *IBM_CUSCO_GROUND_STATES_LIST.pkl* and *RIGETTI-ASPEN-M-3_GROUND_STATES_LIST,pkl *
 
    * **COHERENCE_WITNESS_CIRCUITS_PATH_IBM** and **COHERENCE_WITNESS_CIRCUITS_PATH_RIG**: files with states used to compute Coherence Strength, on Google drive linked above *IBM_CUSCO_COHERENCE_WITNESS_CIRCUITS.PKL* and *RIGETTI-ASPEN-M-3_COHERENCE_WITNESS_CIRCUITS.PKL*

    * **FILE_NAME_HAMILTONIANS_IBM** and **FILE_NAME_HAMILTONIANS_RIG** iles with storing Hamiltonians used in benchmarks, on Google drive linked above *IBM_CUSCO_HAMILTONIANS_DICTIONARY.pkl* AND *RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.pkl*

    * **FILE_NAME_MARGINALS_IBM** and **FILE_NAME_MARGINALS_RIG**: Optionally pre-processed files with 2-qubit marginal data can be loaded, on Google drive linked above *IBM_CUSCO_HAMILTONIANS_DICTIONARY.pkl* AND *RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.pkl*. This data can by omitted and, after suitable modification of the code, recomputed in the script


4. Specify output path where the results will be saved

5. Run the script **article_data_analysis.py** - or jupyter notebook (link coming soon). By default the script performs analysis experimental data for IBM an Rigetti. Note that generation of Fig 4. is by default disabled, since it requires [Manim package](https://www.manim.community/), which is cumbersome to set up and for purpose of this article was run on Windows machine.

If you would run into any issue with running the script or any suggestions, don't hesitate to contact us at [nisq.devices@cft.edu.pl](mailto:nisq.devices@cft.edu.pl).


## Dependencies

For **qrem** package to work properly, the following libraries should be present (and will install if you install via pip):

* "numpy >= 1.18.0, < 1.24",
* "scipy >= 1.7.0",
* "tqdm >= 4.46.0",
* "colorama >= 0.4.3",
* "qiskit >= 0.39.4",
* "networkx >= 0.12.0, < 3.0",
* "pandas >= 1.5.0",
* "picos >= 2.4.0",
* "qiskit-braket-provider >= 0.0.3",
* "qutip >= 4.7.1",
* "matplotlib >= 3.6.0",
* "seaborn >= 0.12.0",
* "sympy >= 1.11.0",
* "pyquil >= 3.0.0",
* "pyquil-for-azure-quantum",
* "ipykernel >= 6.1.0",
* "configargparse >= 1.5.0",
* "python-dotenv >= 1.0.0",

## Optional dependencies

Dependecies for visualizations:

* "manim >= 0.17.2"
