# lunar-free-return

Trajectory optimization code to construct a lunar free-return trajectory. MATLAB and Python version.

## External dependencies
* Python version:
  - NumPy 
  - Matplotlib
  - Scipy
  - extensisq: https://github.com/WRKampi/extensisq  
  - SpiceyPy: https://spiceypy.readthedocs.io/en/stable/

* MATLAB version:
  - MATLAB R2021a or later.

## Required input files
* For SpiceyPy (Python version): Meta-Kernel file "spice.mkn", and Kernel files listed there 
  (see https://spiceypy.readthedocs.io/en/stable/other_stuff.html#lesson-1-kernel-management-with-the-kernel-subsystem)
* For MICE (MATLAB version): Meta-Kernel file "spice.mkn", and Kernel files listed there
  (see https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/MATLAB/index.html)
* Naif SPICE Kernel files downloaded from: https://naif.jpl.nasa.gov/naif/data_generic.html 

## Contributors
Federico Spada
