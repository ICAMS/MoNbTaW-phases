# Calculation of phase composition of MoNbTaW multiprinciple elements alloy

This is a script to compute the phase composition of **MoNbTaW** multiprinciple elements alloy at given temperatures and composition. It requires the list of candidate ordered phases (in *data folder*) with DFT computed energies, volumes, bulk modulus and B'.

There is a possibility to estimate the uncertainty of the phase diagrams by inclusion of a random noise with controlalble magnitude into data.

# Authors

* Yury Lysogorskiy (1) 
* Alberto Ferrari (1,2)

Affiliations:
1. Interdisciplinary Centre for Advanced Materials Simulation, Ruhr-University Bochum, 44801 Bochum, Germany
2. Materials Science and Engineering, Delft University of Technology, 2628CD Delft, The Netherlands

# License

See `LICENSE` file


# Installation

You will require some packages to run the code. These packages could be installed in a following manner.

* installation with `conda`:

```
conda env create -f environment.yml
conda activate HEA
```

* installation with `pip`:
```
pip install ase numpy scipy pandas numba pyyaml  
```

Installation was tested on Python 3.7 and 3.8.

# Example of input files

File name: `input.yaml`
```yaml
# nominal concentration to simulate
nominal_concentration: {Mo: 0.25 , Nb: 0.25, Ta: 0.25, W: 0.25} 

# inclusion of ordered structures based on distance to conv_hull (eV)
max_distance_to_convex_hull: 0.002

# list of temperatures to consider
Ts: [50, 300, 1200]

# number of initial configuration guesses
initial_configuration_guesses: 10

# number of noise-attempts. Set it to 0 if you do not want to apply noise
N_noise_attempts: 0      
# gaussian noise std for disordered phase energies (eV)
disord_energy_std: 0.003  
 # gaussian noise std for ordered phase energies (eV)
ord_energy_std: 0.002    

# number of parallel process to compute independend configuration attempts
max_workers: 2            
# optmizer maxiter
maxiter: 20              
# path to the data folder
DATA_PATH: ./data         
```

# Usage

The script requires the input file `input.yaml`

```
python hea_phase_simulations.py
```


# Output

As a result, the file `total_data.tsv` will be generated. 
This is table, that for each temperature, initial phase configuration and noise attempt contains:
* `T`: temperature (K), 
* `Energy`: energy of the phase configuration (eV/atom)
* `noise_attempt`: index of the noies attempt (=0 for no noise)
* `Disord_phase_1`, ... : Concentrations of disordered phases (from 1 to 4)
* `Phase_1/Mo`, ...  : Composition of each disordered phase (Mo, Nb, Ta, W for each of the four phases )
* `TaMo2W_1018282`, ... : concentration of the given ordered phases

The table is ordered by `Energy`. If no noise were applied, you could consider the minimal energy for each temperature. 
Otherwise, for each `T` and `noise_attempt` minimal energy should be considered. Then these phases results should be averaged over noise attempts.