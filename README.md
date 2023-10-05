# neural-field-synaptic-depression
We implement a neural feild model with non-linear negative feedback in the form of biologically plausible synaptic depression. We use this model to conduct a number of simulations involving the stimulus-response function and traveling pulse entrainment to stimlui. The details will be published shortly.

The published article has some notational differences with the code. First, the code includes a timescale parameter `mu` for the activity variable (`u` equation). In all experiments, this parameter is set to `mu = 1.0`, and in the paper we specifiy that the activity timescale has been normalized and we simply remove it. The synaptic efficacy timescale is the parameter `alpha` in the code, whereas in the paper we use `tau_q`. Lastly, when we begain the project the synaptic depression rate `beta` acounted for the synaptic depression timescale. That is, we had the synaptic efficacy evolution governed by `alpha*q' = 1 - q - alpha*beta*q*f[u]` and we changed it to `alpha*q' = 1 - q - beta*q*f[u]`. All of the numerical calculations use the new model, but some of the symbolic calculations done in jupyter notebooks (for the LaTeX rendering) use the old equations and were not updated. We simply used their resulting formulae and manually replaced `beta -> beta/alpha`.

# Repository Structure
The directory `neural_field_synaptic_depression` is a python module. If you would like to reuse the code for your own purposes, I recommend adding it to your `PYTHON_PATH`. The module is comprised of several sub-modules. The `space_domain` and `time_domain` modules contain objects used to specify the spatial and temporal discretizations. The `neural_field` module uses these dicretizations to generate a forcing term for the resulting system of ODEs (method of lines). The module `time_integrator` is a general purpose ODE solver that heavily relies on python generators, rather than the `scipy` framework. The module `single_step_modifier` extends the `time_integrator` module to analytically evaluate delta forcing terms, and the `time_integrator_tqdm` extends `time_integrator` to include progress pars from the `tqdm` package. Lastly, `root_finding_helpers` includes some utility functions for finding threshold crossings for the non-smooth traveling wave solutions.

Drivers for the `neural_field_synaptic_depression` module can be found in the `front_experiments` and `pulse_experiments` directories. These contain scripts that create animations, run parameter sweeps, and generate the figures for the paper. Here is a list of scripts used to generate data and plots for the figures in the paper.
 - Figure 1.
 	- `front_experiments/figure1.py`
 - Figure 2.
 	- `pulse_experiments/data_speed_width_bifurcation.py`
	- `pulse_experiments/figure2.py`
 - Figure 3.
 	- `front_experiments/spatially_homogeneous_limit.py`
	- `front_experiments/figure3.py`
 - Figure 4.
 	- `pulse_experiments/data_localized.py`
 	- `pulse_experiments/data_localized_q.py`
	- `pulse_experiments/figure4.py`
 - Figure 5.
 	- `pulse_experiments/data_entrainment_square.py`
	- `pulse_experiments/figure5.py`
 - Figure 6.
 	- `pulse_experiments/data_figure6.py`
	- `pulse_experiments/figure6.py`


# Example Driver
The script `example_driver.py` demonstrates basic usage of the package to run simulations.
