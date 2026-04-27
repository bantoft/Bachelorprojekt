#!/bin/sh

### =========================
### General settings
### =========================

#BSUB -J PINN_tr
#BSUB -W 24:00

# ## Output og error logs
# ## -o/-e append, -oo/-eo overwrite
#BSUB -o logs/Output_%J.out
#BSUB -e logs/Output_%J.err

# ## Email notifications
#BSUB -u s224041@dtu.dk
#BSUB -B
#BSUB -N


### =========================
### CPU settings
### =========================


# ## Alle cores på samme node
#BSUB -R "span[hosts=1]"

# ## RAM per core
#BSUB -R "rusage[mem=4GB]"

# ## Kill job hvis den overstiger memory per core
#BSUB -M 5GB


### =========================
### GPU settings
### =========================

# ## Antal GPU cores
#BSUB -n 4

# ## GPU queue
#BSUB -q gpul40

# ## Antal GPU'er
#BSUB -gpu "num=1:mode=exclusive_process" 


### =========================
### Run commands
### =========================

.venv/bin/python SciML/træning.py