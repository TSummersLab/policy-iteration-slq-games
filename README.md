# policy-iteration-slq-games
## Policy iteration for linear quadratic games with stochastic parameters

The code in this repository implements model-based and model-free algorithms of policy iteration for linear quadratic games with stochastic parameters (SLQ games) and ideas from our paper:

"Policy Iteration for Linear Quadratic Games with Stochastic Parameters"
* [IEEE Xplore (IEEE L-CSS, presented @ CDC 2020)](https://ieeexplore.ieee.org/document/9115001)


## Dependencies
* Python 3.5+ (tested with 3.7.3)
* NumPy
* SciPy
* Matplotlib

## Installing
Currently there is no formal package installation procedure; simply download this repository and run the Python files.

## General code structure
The core model parameter estimation code is located in "policy_iteration.py". Various simulated experiments can be run by the functions in "experiments.py". Example linear dynamic system definitions are located in the "problem_data" folder and can be generated using functions in "problem_data_gen.py". Utility functions are located in "matrixmath.py", "extramath.py", and "data_io.py".

## Examples
There are several example experiments which can be run from "experiments.py".

### model_based_robust_stabilization_experiment()
This was used to run the experiment and generate the table presented in the paper.

### model_free_network_slq_game_experiment()
This experiment function solves the SLQ game approximately using policy iteration for a diffusion network system.

### generic_experiment()
This function can be used to try out policy iteration on various SLQ game problems.


## Authors
* **Ben Gravell** - [UT Dallas](https://sites.google.com/view/ben-gravell/home)
* **Karthik Ganapathy** - [UT Dallas](http://www.utdallas.edu/~tyler.summers/)
* **Tyler Summers** - [UT Dallas](http://www.utdallas.edu/~tyler.summers/)
