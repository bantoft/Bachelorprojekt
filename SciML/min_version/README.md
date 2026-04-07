# min_version

This folder contains a compact training setup for the HESEL PINN experiment.

## Structure

- `træning.py`: trains the four PINN models and saves losses to `results/training_history.pt`
- `visualize_training.py`: creates loss plots from the saved history
- `loss_data.py`: loads the HESEL dataset from `BOUT.dmp.0.nc`
- `loss_PDE.py`, `operators.py`, `model.py`, `params2.py`: physics and model helpers
- `results/training_history.pt`: saved training history
- `results/plots/`: generated plots

## Run

Train:

```bash
./bsc_venv/bin/python SciML/min_version/træning.py
```

Visualize:

```bash
./bsc_venv/bin/python SciML/min_version/visualize_training.py
```