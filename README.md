Health-Insurnace-Cross-Sell-Prediction
==============================


## Authors

- [@dev-hack95](https://www.github.com/dev-hack95)

## Project Status
- Complete

## Table of Contents

  - [Problem Statement](#Problem-Statement)
  - [Data source](#data-source)
  - [Methods](#methods)
  - [Tech Stack](#tech-stack)
  - [Quick glance at the results](#quick-glance-at-the-results)
  - [Run Locally](#run-locally)
  - [Explore the notebook](#explore-the-notebook)
  - [Deployment](#Deployment)
  - [Docker](#Docker)
  - [Kubernetes](#Kubernetes)
  - [Repository structure](#repository-structure)
  - [Results](#Results)
  
## Problem Statement
  - Create a machine learning model to predict the insurance defaulters

## Methods

- Exploratory data analysis
- Bivariate analysis
- Multivariate correlation
- Explainable AI
- Model Comparison
- Model deployment

## Tech Stack

- Python (refer to requirement.txt for the packages used in this project)
- Tensorboard
- Javascript
- Docker
- Kubernetes
- DVC
- CML
- Github actions

## Run Locally

1) Initialize git

```bash
git init
```


2) Clone the project

```bash
git clone -b dev-bac https://github.com/dev-hack95/health_insurance_cross_sell_prediction
```

3) enter the project directory

```bash
cd health_insurance_cross_sell_prediction
```

4) install the requriments

```bash
pip install -r requirements.txt
```

5) run application

```bash
streamlit run app.py
```

## RUN on docker


1) Clone the project

```bash
git clone -b dev-bac https://github.com/dev-hack95/health_insurance_cross_sell_prediction
```

2) enter the project directory

```bash
cd health_insurance_cross_sell_prediction
```

3) Build Docker image

```bash
docker build -f ./Dockerfile . -t myapp:latest
```

4) Run docker-compose
```bash
docker-compose up -d
```

5) Stop container
```bash
docker-compose down
```

## Repository structure
Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
