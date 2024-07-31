# Facial Emotion Detection with DenseNet

This repository contains the project work for the module Machine Learning for Physicists 2024. The project focuses on implementing a DenseNet architecture for facial emotion detection and comparing its performance with a traditional Convolutional Neural Network (CNN) approach.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the environment and run the project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/RnLe/MachineLearning.git
    cd MachineLearning
    ```

2. Create and activate the conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate tf_gpu2
    ```

3. Download the dataset from [Google Drive](https://drive.google.com/file/d/1EohQa17A_wiTfE_q6QmBumhSZ0wDhx5_/view?usp=sharing) and extract the `.zip` in the folder `emotions_facial`.

## Usage

To execute the code, you can simply run the `main.ipynb` notebook. This notebook uses the optimal hyperparameters from the `optuna.db` file.

## Files

- `main.ipynb`: Main notebook, using optimal hyperparameters from `optuna.db`.
- `densenet.py`: Implementation of the DenseNet class.
- `alternativeMethod_CNN.ipynb`: Notebook for the alternative method.
- `hyperparam_optimization.py`: Hyperparameter optimization using Optuna.

Note: The optimal hyperparameters are already provided in the project files.

## Acknowledgments

This project was completed as part of the Machine Learning for Physicists 2024 module at [TU Dortmund](https://www.tu-dortmund.de/en/).

For any questions or issues, please feel free to contact the authors.

---

Enjoy exploring the project and feel free to contribute or provide feedback!
