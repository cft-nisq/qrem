# QREM - Quantum Readout Errors Mitigation

This code accompanies the paper ["Efficient reconstruction, benchmarking and validation of cross-talk models
in readout noise in near-term quantum devices"](arxiv link). In order to perform analysis the following steps needed to be taken:

1. Clone the repository, create virtual environment, install development version of the QREM package

2. Download experimental data available [here](https://drive.google.com/drive/folders/14Jh3gJUbiipVLVoWSugJ4uYcZpZWd9XS?usp=drive_link) 

3. Set paths to experimental data in the article_data_analysis.py:

    * DATA_DIRECTORY: directory with experimental data 

    * FILE_NAME_RESULTS_IBM and FILE_NAME_RESULTS_RIG: files with raw experimental results, on Google drive linked above RESULTS_IBM_CUSCO.pkl and RESULTS_RIGETTI_ASPEN-M-3.pkl

    * FILE_NAME_GROUND_STATES_LIST_IBM and FILE_NAME_GROUND_STATES_LIST_RIG: files with ground states used in benchmarks, on Google drive linked above   IBM_CUSCO_GROUND_STATES_LIST.pkl and RIGETTI-ASPEN-M-3_GROUND_STATES_LIST,pkl 
 
    * COHERENCE_WITNESS_CIRCUITS_PATH_IBM and COHERENCE_WITNESS_CIRCUITS_PATH_RIG: files with states used to compute Coherence Strength, on Google drive linked above IBM_CUSCO_COHERENCE_WITNESS_CIRCUITS.PKL and RIGETTI-ASPEN-M-3_COHERENCE_WITNESS_CIRCUITS 

    * FILE_NAME_HAMILTONIANS_IBM and FILE_NAME_HAMILTONIANS_RIG iles with storing Hamiltonians used in benchmarks, on Google drive linked above IBM_CUSCO_HAMILTONIANS_DICTIONARY.pkl AND RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.pkl

    * FILE_NAME_MARGINALS_IBM and FILE_NAME_MARGINALS_RIG: Optionally pre-processed files with 2-qubit marginal data can be loaded, on Google drive linked above IBM_CUSCO_HAMILTONIANS_DICTIONARY.pkl AND RIGETTI-ASPEN-M-3_HAMILTONIANS_DICTIONARY.pkl. This data can by omitted and, after suitable modification of the code, recomputed in the script


4. Specify path where the results will be saved 

5. Run the script article_data_analysis.py. By default the script performs analysis experimental data for IBM an Rigetti. Note that generation of Fig 4. since it requires Manim package, and is by default disabled.
