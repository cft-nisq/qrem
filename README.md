# QREM - Quantum Readout Errors Mitigation

This package provides a versatile set of tools for the characterization and mitigation of readout noise in NISQ devices. Standard characterization approaches become infeasible with the growing size of a device, since the number of circuits required to perform tomographic reconstruction of a measurement process grows exponentially in the number of qubits. In QREM  we use efficient techniques that circumvent those problems by focusing on reconstructing local properties of the readout noise.

You can find article based on initial version of this package [here - http://arxiv.org/abs/2311.10661](http://arxiv.org/abs/2311.10661) and the corresponding code used at the moment of writning the article [here](https://github.com/cft-nisq/qrem/tree/article-eff).

<img src="https://quantin.pl/wp-content/uploads/2023/03/washington_26_04_2022_worst_case_classical_threshold_2.png"
  alt="Plot of correlation coefficients determined in characterization on device layout"
  title="Plot of correlation coefficients determined in characterization on device layout"
  style="display: inline-block
  margin: 0 auto
  max-width: 400px"/>

## Status of development

This package is released now as an alpha version, to gather feedback while it undergoes final adjustments prior to the first release. As it is under heavy development, existing functionalities might change, while new functionalities and notebooks are expected to be added in the future.

## Documentation

Current documentation (work in progress) is available [here](https://cft-nisq.github.io/qrem/index.html)

## Introduction

The two current main functionalities are:

### **Noise characterization**

* experiment design
* hardware experiment implementation and data processing (on devices supported by qiskit/pyquil)
* readout noise characterisation
* learning of noise models

### **Noise mitigation**

* mitigation based on noise model provided by user ( currently available is CN, CTMP is under development)

## Installation

The best way to install this package is to use pip (see [pypi website](https://pypi.org/project/qrem/)):

```console
pip install qrem
```

This method will automatically install all required dependecies (see [below for list of dependecies](#dependencies)).

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
  
## References

**The workflow of this package is mainly based on works**:
  
[1] Filip B. Maciejewski, Zoltán Zimborás, Michał Oszmaniec, "Mitigation of readout noise in near-term quantum devices by classical post-processing based on detector tomography", [Quantum 4, 257 (2020)](https://quantum-journal.org/papers/q-2020-04-24-257/)

[2] Filip B. Maciejewski, Flavio Baccari, Zoltán Zimborás, Michał Oszmaniec, "Modeling and mitigation of cross-talk effects in readout noise with applications to the Quantum Approximate Optimization Algorithm", [Quantum 5, 464 (2021)](https://quantum-journal.org/papers/q-2021-06-01-464/)

**Further references:**

[3]. Sergey Bravyi, Sarah Sheldon, Abhinav Kandala, David C. Mckay, Jay M. Gambetta, Mitigating measurement errors in multi-qubit experiments, [Phys. Rev. A 103, 042605 (2021)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.042605)
  
[4]. Flavio Baccari, Christian Gogolin, Peter Wittek, and Antonio Acín, Verifying the output of quantum optimizers with ground-state energy lower bounds, [Phys. Rev. Research 2, 043163 (2020)](https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.2.043163)
