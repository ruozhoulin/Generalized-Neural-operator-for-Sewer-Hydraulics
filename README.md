# Generalized Neural operator for Sewer Hydraulics

This repository contains the code, trained model parameters, and example data accompanying a manuscript currently under submission.

## Contents

- `tutorial.ipynb`: tutorial notebook for model prediction, evaluation, and visualization.
- `operators.py`: neural operator implementations.
- `utilities.py`: data processing, evaluation, and plotting helpers.
- `input/`: trained model parameters, example dataset, and sewer network file.

## Data sources

- **Sewer network data:** https://data.gov.hk/en-datasets
- **Rainfall data:** https://data.weather.gov.hk/weatherAPI/opendata/hourlyRainfall.php?lang=en

## Setup and execution

The development environment uses **Python 3.12.9**. Package versions are listed in [requirements.txt](requirements.txt).

Create and activate a Python 3.12.9 environment, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The original environment uses PyTorch 2.7.0 with CUDA 12.8. The requirements specify the PyTorch release version; the appropriate build depends on your platform.

## License and citation

License and citation information will be updated after publication.
