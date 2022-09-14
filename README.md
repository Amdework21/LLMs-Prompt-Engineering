
<h1 align="center">LLMs-Prompt-Engineering</h1>
<div>
<a href="https://github.com/Amdework21/LLMs-Prompt-Engineering/"><img src="https://img.shields.io/github/forks/Amdework21/LLMs-Prompt-Engineering" alt="Forks Badge"/></a>
<a href="https://github.com/Amdework21/LLMs-Prompt-Engineering/pulls"><img src="https://img.shields.io/github/issues-pr/Amdework21/LLMs-Prompt-Engineering" alt="Pull Requests Badge"/></a>
<a href="https://github.com/Amdework21/LLMs-Prompt-Engineering/issues"><img src="https://img.shields.io/github/issues/Amdework21/LLMs-Prompt-Engineering" alt="Issues Badge"/></a>
<a href="https://github.com/Amdework21/LLMs-Prompt-Engineering/graphs/contributors"><img alt="GitHub contributors" src="https://img.shields.io/github/contributors/Amdework21/LLMs-Prompt-Engineering?color=2b9348"></a>
<a href="https://github.com/Amdework21/LLMs-Prompt-Engineering/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Amdework21/LLMs-Prompt-Engineering?color=2b9348" alt="License Badge"/></a>
</div>

</br>

![LLM-image](https://i0.wp.com/bdtechtalks.com/wp-content/uploads/2022/06/large-language-model-logical-reasoning.jpg?ssl=1)

## API link
- [visualization link](###)

## Articles
- [Medium Article](###)

## Table of Contents

* [LLMs-Prompt-Engineering](#LLMs-Prompt-Engineering)

  - [Introduction](##Introduction)
  - [Project Structure](#project-structure)
    * [data](#data)
    * [models](#models)
    * [notebooks](#notebooks)
    * [scripts](#scripts)
    * [sql](#sql)
    * [tests](#tests)
    * [logs](#logs)
    * [root folder](#root-folder)
  - [Installation guide for windows](#installation-guide-for-windows)
  - [Installation guide for linux](#installation-guide-for-linux)

## Introduction
Large Language Models coupled with multiple AI capabilities are able to generate images and text, and also approach/achieve human level performance on a number of tasks.The world is going through a revolution in art (DALL-E, MidJourney, Imagine, etc.), science (AlphaFold), medicine, and other key areas, and this approach is playing a role in this revolution. This project is then created to use some of the above LLM APIs to compare web pages in relation with Job description.

## Project Structure

### images:

- `images/` the folder where all snapshot for the project are stored.

### logs:

- `logs/` the folder where script logs are stored.

### mlruns:
- `mlruns/0/` the folder that contain auto generated mlflow runs.
### data:

 - `train_store.csv.dvc` the folder where the dataset versioned csv files are stored.

### .dvc:
- `.dvc/`: the folder where dvc is configured for data version control.

### .github:

- `.github/`: the folder where github actions and CML workflow is integrated.

### .vscode:

- `.vscode/`: the folder where local path fix are stored.
### modles:
- `llmodel.pkl`: the folder where model pickle files are stored.

### notebooks:

- `data_preProcessing.ipynb`: a jupyter notebook for preprocessing the data.
- `data_exploration.ipynb`: a jupyter notebook for exploring the data.
- `ml_preProcess`: a jupyter notebook for preprocessing the data for ml analysis.
- `ml_model`: a jupyter notebook training an Regression models for prediction purpose.
- `nlp_transformer.ipynb`: a jupyter notebook training an LSTM model for forecasting purpose.

###  scripts:

- `data_cleanning_.py`: a python script for cleaning data with pandas dataframes.
- `logger.py`: a python script for creating logs 
- `read_write_util.py`:  a python script for reading and writting files.
- `ltsm_model`: a python script for model manipulation.
- `data_manipulator.py`: a python script for manipulating dataframes.
- `data_exploration.py`: a python script for plotting dataframes.
- `multiapp.py`: a python script for creating a multipaged streamlit app.
- `log_help.py`: a python script that creates python based logger.
### tests:

- `tests/`: the folder containing unit tests for the scripts.

### sql:

- `sql/`: the folder containing database table and mysql-python manipulator script.
### root folder

- `train.py`: holds cml report and model metrics.
- `results.txt`: contains cml pr reports.
- `requirements.txt`: a text file lsiting the projet's dependancies.
- `.travis.yml`: a configuration file for Travis CI for unit test.
- `app.py`: main file for the streamlit application.
- `setup.py`: a configuration file for installing the scripts as a package.
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.
- `Dockerfile`: build users can create an automated build that executes several command-line instructions in a container.

## Installation guide for windows

```bash
git clone https://github.com/Amdework21/LLMs-Prompt-Engineering.git
cd LLMs-Prompt-Engineering
pip install python3 setup.py
```
## Installation guide for Linux

```bash
git clone https://github.com/Amdework21/LLMs-Prompt-Engineering.git
cd LLMs-Prompt-Engineering
sudo python3 setup.py install
```
