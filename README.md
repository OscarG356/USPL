# USPL Injection Current Estimation using Machine Learning

## Overview

This project implements a machine learning pipeline to estimate the **injection current (Ip)** of a graphene-based ultrashort-pulse fiber laser (USPL) using only photodetected electrical signals.

The approach replaces traditional optical instrumentation (e.g., OSA) with a data-driven method based on multi-domain feature extraction and regression models.

The methodology and results are based on the work presented in the accompanying research paper.

---

## Key Features

* Multi-domain feature analysis:

  * Temporal
  * Statistical
  * Spectral

* Multiple regression models:

  * Support Vector Regression (SVR)
  * Random Forest (RF)
  * Bayesian Ridge (BR)
  * XGBoost (XGB)

* Model interpretability:

  * SHAP values
  * Permutation importance

* Iterative training pipeline with:

  * Cross-validation
  * Performance tracking
  * Convergence analysis

* Automatic experiment tracking:

  * Each run is stored in a timestamped directory

---

## Project Structure

```
project/
│── src/                    # Core ML pipeline
│── data/
│   ├── raw/               # Raw data (not tracked)
│   ├── processed/         # Feature datasets
│   └── outputs/           # Experiment results (timestamped runs)
│── reports/
│   └── figures/           # Final visualizations
│── legacy/                # Old versions
│── requirements.txt
│── README.md
```

---

## Installation

1. Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

---

## Usage

Run the main pipeline:

```bash
python src/temporallearning_opnet.py
```

---

## Input Data

The pipeline expects the following file:

```
data/processed/extracted_features.csv
```

### Required format:

* Each row: one signal sample
* Columns: extracted features
* Target column: `target_current`

---

## Outputs

Each execution creates a new directory:

```
data/outputs/run_YYYYMMDD_HHMMSS/
```

This directory contains:

* Model predictions
* Feature importance rankings (SHAP & permutation)
* Iteration logs
* Statistical summaries
* Convergence analysis data

Additionally, key plots are copied to:

```
reports/figures/
```

### Generated Visualizations:

* Convergence analysis (ranking & importance)
* Prediction consensus across models
* Feature importance evolution

---

## Methodology Summary

The pipeline:

1. Loads precomputed features
2. Splits data into train/test sets
3. Trains multiple regression models
4. Evaluates performance (R², MAE, RMSE, MAPE)
5. Computes feature importance:

   * Permutation importance
   * SHAP values
6. Aggregates results across iterations
7. Generates convergence and consensus plots

---

## Reproducibility Notes

* Each run is isolated using timestamped directories
* Results are cumulative and non-destructive
* Randomness comes from train/test splits

---

## Research Context

This implementation is based on the following work:

> *Machine Learning-Based Estimation of Injection Current in an Ultrashort-Pulse Laser Using Photodetected Signal Features*

The method demonstrates that injection current can be estimated with high accuracy using only electrical signal features, avoiding the need for optical spectrum analysis.

---

## Author

Oscar A. Gutierrez Rivadeneira
Universidad de Antioquia – GITA Lab

---

## License

This project is intended for academic and research purposes.
