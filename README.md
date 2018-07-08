# Bachelor thesis code

Title:Exploring the Relation Between Thermospheric
Density and Field Aligned Currents (FAC) Based on Swarm Satellite Observations

Author: Simon Kok Lupemba

Contact: simon.kok@live.dk

This folder contain the code used for processing Swarm data and plotting of the
results. This is not a complete toolbox and there are several location where
file paths are hard coded. The focus of this project have not been programming
and the code should be seen as a tool and not as the final product.
This repository serves to document the code used in the project.

Feel free to use part of the code for other project.

Dependencies:numpy, pandas, apexpy, pyamps, spacepy, swarmtoolkit and
matlab aerospace toolbox (only for density_normalization.m)

The swarmtoolkit is only used to read the cdf files from swarm and if I where
to write the code again I would only use spacepy.
