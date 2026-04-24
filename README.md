# ve-composite

This repository contains python code for modeling viscoelastic composites of dilute suspensions.

A composite consisting of a viscoelastic matrix containing a small volume fraction of viscoelastic inclusions (idealized as disperse spheres or randomly oriented platelets) can be approximated as a homogeneous, isotropic material with linear viscoelastic properties dependent on those of the constituents. In this repository, we handle the case of dilute viscoelastic inclusions in a viscoelastic matrix.

Classes/functions are defined in `composite_tools.py`, and usage of the `SphericalInclusion` class is demonstrated in `startup.ipynb`. The usage of `PlateletInclusion` is similar.

### Startup

The notebook `startup.ipynb` can be viewed in the web browser with precomputed output. To run the notebook with python and standard packages (numpy, matplotlib) installed, open and run `startup.ipynb`. To use the composite modulus functions, include `composite_tools.py`.
