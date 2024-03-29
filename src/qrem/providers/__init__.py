"""
Quantum Experimentation Providers Subpackage 
============================================

This subpackage (qrem.providers) contains modules that provide functionalities for interacting with various quantum computing platforms and simulators. It enables users to run characterization, mitigation, and benchmarking experiments on different quantum devices and simulators. The subpackage is organized into modules, each catering to a specific quantum computing provider or simulation environment.

Modules
-------
    AWS-Braket Provider Module (qrem.providers.aws_braket)
        Facilitates experiments on AWS Braket devices. This module includes functions for device property retrieval, circuit translation to Braket format, circuit preparation, and execution on AWS Braket services.

    IBM Provider Module (qrem.providers.ibm)
        Offers functionalities for conducting experiments on IBM Quantum machines. It allows for connecting to IBM quantum backends, querying backend properties, translating and executing circuits in Qiskit format, and handling experiment results. This module interacts with IBM Quantum services via QiskitRuntimeService and/or IBMServiceBackend.

    Simulation Module (qrem.providers.simulation)
        Provides tools for simulating the results of quantum computer characterization experiments. This module supports simulations with clean and noisy quantum computer models, utilizing 'cn' and 'ctmp' (coming soon) noise models for realistic quantum noise simulation.

Usage
-----
    Each module within this subpackage is tailored to specific quantum computing services or simulation needs. Users can leverage these modules to execute quantum experiments, retrieve device properties, and simulate quantum computations on various platforms. This subpackage serves as a bridge between the QREM library and external quantum computing platforms and simulators, offering a unified interface for various quantum experiments.

Note
----
    To use any of these modules, appropriate credentials and access to the respective quantum computing services or simulation environments are required. Users should ensure they have the necessary setup and permissions for each platform they intend to interact with - you can use .env file placed in your project folder used for this purpose. For more information, refer to the documentation for each module.
"""