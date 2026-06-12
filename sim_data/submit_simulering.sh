#!/bin/sh

### =========================
### General settings
### =========================

# ##  queue
#BSUB -q hpc

#BSUB -J Plsma_sim
#BSUB -W 6:00

# ## Output og error logs
# ## -o/-e append, -oo/-eo overwrite
#BSUB -o Bachelorprojekt/sim_data/logs/Output_Alexander__grid_c_2.out

#BSUB -e Bachelorprojekt/sim_data/logs/Output_Alexander__grid_c_2.err

# ## Email notifications
# ## BSUB -u s224041@dtu.dk
# ## BSUB -B

# ##BSUB -N


### =========================
### CPU settings
### =========================

# ## Antal CPU cores
#BSUB -n 16

# ## Alle cores på samme node
#BSUB -R "span[hosts=1]"

# ## RAM per core
#BSUB -R "rusage[mem=4GB]"

# ## Kill job hvis den overstiger memory per core
#BSUB -M 5GB


### =========================
### Run commands
### =========================

module load mpi/5.0.3-gcc-13.3.0-binutils-2.42
module load fftw3/3.3.10-openmpi-5.0.3-gcc-13.3.0
module load netcdf-c/4.9.3-hdf5-1.14.6-intel-2021-update4
module load intel/2021.4.0


cd BOUT-HESEL

mpirun -np 16 ./hesel -d Alexander__grid_c_2
