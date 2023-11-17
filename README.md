# QREM - Quantum Readout Errors Mitigation

This code accompanies the paper ["Efficient reconstruction, benchmarking and validation of cross-talk models
in readout noise in near-term quantum devices"](arxiv link). In order to perform analysis the following steps needed to be taken:

1. Clone the repository, create virtual environment, install development version of the QREM package

2. Download experimental data available [here](https://drive.google.com/drive/folders/14Jh3gJUbiipVLVoWSugJ4uYcZpZWd9XS?usp=drive_link) 

3. Set paths to experimental data in the article_data_analysis.py

5. Specify path where the results will be saved 

4. Run the script article_data_analysis.py. By default the script performs analysis experimental data for IBM an Rigetti. Note that generation of Fig 4. since it requires Manim package, and is by default disabled.