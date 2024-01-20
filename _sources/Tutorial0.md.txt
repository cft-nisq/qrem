# 0. Installation

Quantum computers of today, categorized as noisy intermediate-scale quantum (NISQ) devices, face a significant challenge due to readout errors. These errors substantially contribute to the overall noise affecting quantum computations. For devices with a large number of qubits, fully characterizing generic readout noise is infeasible.

QREM software package is specifically designed to characterize and mitigate readout errors in large-scale quantum devices. QREM focuses on characterisation and mitigation of correlated and non-local errors, which are prevalent in these systems.

This tutorial series is intended to guide users through the fundamental workflow of the QREM package. Our aim is to provide a clear, step-by-step approach to help you effectively utilize QREM for enhancing quantum computing operations.

We will begin by detailing the steps to install the QREM package, covering both the standard and developer modes of installation. In this first part we will cover:

## Table of Contents

- [0. Installation](#0-installation)
  - [Table of Contents](#table-of-contents)
  - [Links](#links)
  - [Installation](#installation)
  - [Installation in editable mode from source](#installation-in-editable-mode-from-source)
  - [Dependencies](#dependencies)
  - [Optional dependencies](#optional-dependencies)

## Links

1. PyPI QREM package page: [https://pypi.org/project/qrem/](https://pypi.org/project/qrem/)
2. Source code on GitHub: [https://github.com/cft-nisq/qrem](https://github.com/cft-nisq/qrem)
3. Documentation: [https://cft-nisq.github.io/qrem/index.html](https://cft-nisq.github.io/qrem/index.html)
4. Articles: [Tuziemski, Jan, et al. "Efficient reconstruction, benchmarking and validation of cross-talk models in readout noise in near-term quantum devices." arXiv preprint arXiv:2311.10661 (2023).](http://arxiv.org/abs/2311.10661)

## Installation

The best way to install this package is to use pip. We suggest however, to create a separate virtual enviroment for work with qrem using venv.

1. Make sure that you have Python newer than 3.9.0 and newest version of pip. If not - best to upgrade both before proceeding further.

2. Create and go to your desired installation directory for the project. Go there and create a new virutal environment:
   1. For windows:

        ```console
        python -m venv venv_qrem    
        ```

   2. For linux:

        ```console
        python3 -m venv venv_qrem
        ```

3. Activate your virtual environment:
    1. For windows:

        ```console
        .\venv_qrem\Scripts\activate
        ```

    2. For linux:

        ```console
        source venv_qrem/bin/activate
        ```

4. Install qrem package from PyPI:

    ```console
    pip install qrem
    ```

    This method will automatically install all required dependecies in the created virtual environment (see [below for list of dependecies](#dependencies)).

5. QREM uses dotenv module to set up all the api keys etc. Create a .env file in your project folder, and set up all the relevant keys and env variables there (use .env-default file as a template, you can find example .env file on the GitHub repo [here](https://github.com/cft-nisq/qrem/blob/master/.env-default)).

## Installation in editable mode from source

1. To install from source, first create virtual environment by following steps 1-3 in [Installation from PIP](#installation) instruction,
2. Clone the repository from github directly into the project directory (mind the dot):

    ```console
    git clone git@github.com:cft-nisq/qrem.git .
    ```

    or

    ```console
    git clone https://github.com/cft-nisq/qrem.git .
    ```

3. Install package in editable mode:
    1. For windows:

        ```console
        pip install --editable .
        ```

    2. For linux:

        ```console
        pip3 install --editable .
        ```

## Dependencies

For **qrem** package to work properly, the following libraries should be present (and will install if you install via pip):

* "configargparse >= 1.5.0",
* "python-dotenv >= 1.0.0",
* "orjson >= 3.9.10",
* "tqdm >= 4.64.0",
* "numpy >= 1.18.0, < 1.24",
* "scipy >= 1.7.0",
* "networkx >= 0.12.0, < 3.0",
* "pandas >= 1.5.0",
* "matplotlib >= 3.6.0",
* "sympy >= 1.11.0",
* "qiskit >= 0.43.3",
* "qiskit-ibm-runtime >= 0.11.2",
* "qiskit-ibm-provider >= 0.6.1",
* "qiskit-braket-provider >= 0.0.3",
* "amazon-braket-sdk >= 1.61.0",
* "pyquil >= 4.0.3",
* "qutip >= 4.7.1",
* "picos >= 2.4.0",
* "seaborn >= 0.12.0",
* "colorama >= 0.4.6",
* "ipykernel >= 6.1.0",

## Optional dependencies

Dependecies for visualizations, currently tested and working only on Windows machines. Visualization module is currently being updated, the dependency may change:

* "manim >= 0.17.2"
  