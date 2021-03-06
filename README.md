# Kaggle House Prices

![Unit Test Coverage](docs/assets/unit-test-badge.svg)

Working through the kaggle house prices playground competition to learn about ML

# TODO list

## Project

[x] Work through pipelines talk: https://www.youtube.com/watch?v=XpOBSaktb6s&list=PL_pVmAaAnxIRnSw6wiCpSvshFyCREZmlM&index=5
[] Understand how transformers/estimators can be used to add a feature in an sklearn pipeline
[] How to get the best result from GridSearchCV
[] Add linear regression model with full feature processing pipeline
[] Create a way to put a kaggle submission together from a model
[] Decide on full evaluation metrics and plots to use (can I use yellowbrick?)
[] Add MLFlow instrumentation and logging
[] Add regularised models
    * LASSO
    * Ridge
    * ElasticNET
[] Abstract configuration for models
[] Provide a way to select model via command line parameter
[] Abstract pipeline stages?
    * Data load
    * Pre-processing / feature engineering
        * pre-split feature processing
        * split
        * post-split processing (scaling etc)
    * Model fit/training
    * Store
    * Evaluation
[] Add tests for feature processing
[] Add column names constants


## Cookiecutter todo

[] Remove features/build_features

## Prerequisites

- [Anaconda](https://www.anaconda.com/download/) >=5.x

# Installation guide

## Set up conda environment

Using conda:

```
conda env create -f environment.yml
activate kaggle_house_prices
```

The packages necessary to run the project are now installed inside the conda environment.

**Note: The following sections assume you are located in your conda environment.**

## Set up project's module

To move beyond notebook prototyping, all reusable code should go into the `src/` folder package. To use that package inside your project, install the project's module in editable mode, so you can edit files in the `src/` folder and use the modules inside your notebooks :

```
pip install --editable .
```

To use the module inside your notebooks, add `%autoreload` at the top of your notebook :

```
%load_ext autoreload
%autoreload 2
```

Example of module usage :

```py
from src.data.make_dataset import generate
generate(10)
```

## Set up Git diff for notebooks and lab

We use [nbdime](https://nbdime.readthedocs.io/en/stable/index.html) for diffing and merging Jupyter notebooks.

To configure it to this git project :

```
nbdime config-git --enable
```

To enable notebook extension :

```
nbdime extensions --enable --sys-prefix
```

Or, if you prefer full control, you can run the individual steps:

```
jupyter serverextension enable --py nbdime --sys-prefix

jupyter nbextension install --py nbdime --sys-prefix
jupyter nbextension enable --py nbdime --sys-prefix

jupyter labextension install nbdime-jupyterlab
```

You may need to rebuild the extension : `jupyter lab build`

## Set up Plotly for Jupyterlab

Plotly works in notebook but further steps are needed for it to work in Jupyterlab :

* @jupyter-widgets/jupyterlab-manager # Jupyter widgets support
* plotlywidget  # FigureWidget support
* @jupyterlab/plotly-extension  # offline iplot support

There are conflict versions between those extensions so check the [latest Plotly README](https://github.com/plotly/plotly.py#installation-of-plotlypy-version-3) to ensure you fetch the correct ones. 

```
jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.36 --no-build
jupyter labextension install plotlywidget@0.2.1  --no-build
jupyter labextension install @jupyterlab/plotly-extension@0.16  --no-build
jupyter lab build
```

# Invoke command

We use [Invoke](http://www.pyinvoke.org/) to manage an
unique entry point into all of the project tasks.

List of all tasks for project :

```
$ invoke -l

Available tasks:

  lab     Launch Jupyter lab
```

Help on a particular task :

```
$ invoke --help lab
Usage: inv[oke] [--core-opts] notebook [--options] [other tasks here ...]

Docstring:
  Launch Jupyter lab

Options:
  -i STRING, --ip=STRING   IP to listen on, defaults to *
  -p, --port               Port to listen on, defaults to 8888
```

You will find the definition of each task inside the `tasks.py` file, so you can add your own.

_PS : we don't use Makefile because some people work on Windows workstations and the
install of make is cumbersome on those._

# Project organization

    ????????? tasks.py           <- Invoke with commands like `notebook`
    ????????? README.md          <- The top-level README for developers using this project.
    ????????? data
    ???   ????????? external       <- Data from third party sources.
    ???   ????????? interim        <- Intermediate data that has been transformed.
    ???   ????????? processed      <- The final, canonical data sets for modeling.
    ???   ????????? raw            <- The original, immutable data dump.
    ???
    ????????? models             <- Trained and serialized models, model predictions, or model summaries
    ???
    ????????? notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    ???                         the creator's initials, and a short `-` delimited description, e.g.
    ???                         `1.0-jqp-initial-data-exploration`.
    ???
    ????????? references         <- Data dictionaries, manuals, and all other explanatory materials.
    ???
    ????????? reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    ???   ????????? figures        <- Generated graphics and figures to be used in reporting
    ???
    ????????? environment.yml    <- The requirements file for reproducing the analysis environment
    ???
    ????????? src                <- Source code for use in this project.
        ???kaggle_house_prices
            ????????? __init__.py    <- Makes src a Python module
            ???
            ????????? data           <- Scripts to download or generate data
            ???   ????????? make_dataset.py
            ???
            ????????? features       <- Scripts to turn raw data into features for modeling
            ???   ????????? build_features.py
            ???
            ????????? models         <- Scripts to train models and then use trained models to make
            ???   ???                 predictions
            ???   ????????? predict_model.py
            ???   ????????? train_model.py
            ???
            ????????? visualization  <- Scripts to create exploratory and results oriented visualizations
                ????????? visualize.py
