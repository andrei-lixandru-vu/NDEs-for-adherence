# Adherence Prediction with Neural ODEs

This is a cleaned and modularized version of the neural ODE adherence prediction code.

## Project Structure

- `config.py` - Configuration parameters and constants
- `data_generation.py` - Synthetic data generation functions
- `model.py` - Neural network model architecture
- `datasets.py` - PyTorch Dataset classes
- `losses.py` - Loss functions for training and validation
- `training.py` - Training logic
- `visualization.py` - Plotting utilities
- `main.py` - Main script that orchestrates the experiment

## Running the Code

### From command line:

```bash
python -m clean_ndes.main
```

Or from within the clean_ndes directory:

```bash
cd clean_ndes
python main.py
```

### From notebook:

```python
from clean_ndes import *

# Or import specific components
from clean_ndes.data_generation import generate_synthetic_data_super_simple
from clean_ndes.model import AdherencePredictor
from clean_ndes.training import train_one_epoch, train_multiple_runs

# Run full experiment
from clean_ndes.main import main
main()
```

## Experiment Description

The code compares two sampling strategies:
1. **Intra-trajectory sampling**: Uses sliding windows of length 2 from full trajectories
2. **Inter-trajectory sampling**: Uses full trajectories

Both strategies train the same neural network model to predict adherence given control variables.

## Configuration

Key parameters can be adjusted in `config.py`:
- `N_TRAJECTORIES`: Number of patients
- `T_TIME_STEPS`: Number of days
- `D_DIMENSIONS`: Number of features (adherence + control variables)
- `BATCH_SIZE`: Training batch size
- `NUM_RUNS`: Number of runs for statistics

## Notes

- The original messy notebook has been preserved as `clean_NDEs_original.py` in the parent directory for reference
- All functionality from the original has been preserved and organized into separate modules
- The code is now importable as a Python package from notebooks or scripts
