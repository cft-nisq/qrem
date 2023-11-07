This module allows for visualisation of tomography results.
It has 3 main functionalities:
1. Plotting on layout: basic layout structure, correlations coefficients between qubits (implemented), cluster structure, readout lines (in development)
2. Plotting cluster structure on conceptual graphs (implemented)
3. Ploting histograms of correlation coefficients (implemented)
4. Plotting heatmaps of correlation coefficients (implemented)
 
**For an example of usage see the notebook tutorial_visualisation.**
These functionalities are implemented in file visualisation. File device_constants contains layout information about some devices, e.g. Rigetti's ASPEN-M-1. File plotting_constants contains constants used across the module, like colors etc.

This module requires installation of Manim Community (>= 0.17.2), which also requires ffmeg (here 6.0). LaTeX is needed as well (here we use MiKTeX 22.1). For installation instructions of all 3 see: https://docs.manim.community/en/stable/installation.html.
