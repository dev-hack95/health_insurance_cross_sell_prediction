Health-Insurnace-Cross-Sell-Prediction
==============================


## Authors

- [@dev-hack95](https://www.github.com/dev-hack95)

## Project Status
- Complete (Note: Use dev-bac branch , Bagging Classifer giving heighest accuracy)

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
  - Create a machine learning model on diabetes detection

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

## Quick glance at the results

```bash
{'Logistic_Regression': 0.7835004557885141,
 'Navie_Bayes': 0.7813354603463992,
 'lda': 0.7843740504405956,
 'Random_Forest': 0.9179580674567,
 'Ada_boost': 0.8453737465815861,
 'Gradient_boost': 0.8599969614099058,
 'Bagging_Classifer': 0.9312518991188089,
 'knn_classifier': 0.8704421148587056,
 'Decision_tree': 0.8915223336371924,
 'Extr_tree': 0.9226678821027043}
 ```
 
Top 3 models (with default parameters)

| Model     	                |  score 	          |
|-------------------	        |------------------	|
| Extra Tree Classifier  	    | 92.3% 	          |
| Random Forest    	          | 91.8% 	          |
| Bagging_Classifer           | 93.2% 	          |



## Run Locally

1) Initialize git

```bash
git init
```


2) Clone the project

```bash
git clone -b dev-bac https://github.com/dev-hack95/Diabetes_detection
```

3) Enter the project directory

```bash
cd Diabetes_detection
```

4) Install the requriments

```bash
pip install -r requirements.txt
```
5) DVC

```bash
dvc repro
```

6) Run application

```bash
streamlit run app.py
```

## Explore the notebook

To explore the notebook file [here](https://github.com/dev-hack95/health_insurance_cross_sell_prediction/blob/dev-bac/notebooks/Health%20Insurance%20Cross%20Sell%20Prediction.ipynb)


## Kubernetes

```bash
kubectl -f apply application/deployment.yml
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
    ├── src                <- Source code for use in this project
    │   │
    │   |
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── preprocess.py
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
