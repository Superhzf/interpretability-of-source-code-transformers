import math
import numpy as np
import torch
import torch.nn as nn
import skorch
import os
import glob as glob

from . import utils
from . import metrics

from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV


def save_hyperparam(text, path):
    """
    Function to save hyperparameters in a `.yml` file.
    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    with open(path, 'w') as f:
        keys = list(text.keys())
        for key in keys:
            f.writelines(f"{key}: {text[key]}\n")

def create_run():
    """
    Function to create `run_<num>` folders in the `outputs` folder for each run.
    """
    num_run_dirs = len(glob.glob('../outputs/run_*'))
    run_dir = f"../outputs/run_{num_run_dirs+1}"
    os.makedirs(run_dir)
    return run_dir 

def creat_search_run():
    """
    Function to save the Grid Search results.
    """
    num_search_dirs = len(glob.glob('../outputs/search_*'))
    search_dirs = f"../outputs/search_{num_search_dirs+1}"
    os.makedirs(search_dirs)
    return search_dirs

def save_best_hyperparam(text, path):
    """
    Function to save best hyperparameters in a `.yml` file.
    :param text: The hyperparameters dictionary.
    :param path: Path to save the hyperparmeters.
    """
    with open(path, 'a') as f:
        f.write(f"{str(text)}\n")

\
#Get X_train and y_train values
def grid_search_logreg_model(
    X_train,
    y_train,
):
    search_folder = creat_search_run()

    # Learning parameters.
    lr = 0.001
    num_epochs = 20
    device = 'cpu'
    print(f"Computation device: {device}\n")

    criterion = nn.CrossEntropyLoss()
    # Instance of `NeuralNetClassifier` to be passed to `GridSearchCV`
    net = NeuralNetClassifier(
        module=utils.LinearNet(X_train.shape[1], 2), num_epochs=num_epochs,
        optimizer=torch.optim.Adam,
        criterion=criterion,
        lr=lr, verbose=1,n_jobs=-1
    )
    params = {
        'lr': [0.001, 0.01, 0.0005],
        'num_epochs': list(range(10, 55, 7)),
        'optimizer__lambda_l1': [0.0001, 0.00001, 0.000001, 0.0000001],
        'optimizer__lambda_l2': [0.0001, 0.00001, 0.000001, 0.0000001],

    }
    """
    Define `GridSearchCV`.
    4 lrs * 7 max_epochs * 4 module__first_conv_out * 3 module__first_fc_out
    * 2 CVs = 672 fits.
    """
    #Do not need these
    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    gs = GridSearchCV(
        net, params, refit=False, scoring='f1', verbose=1, cv=3
    )

    gs.fit(X_train,y_train)

    #return (gs.best_Score_,gs.best_params_)
    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
    save_best_hyperparam(gs.best_score_, f"../outputs/{search_folder}/best_param.yml")
    save_best_hyperparam(gs.best_params_, f"../outputs/{search_folder}/best_param.yml")

    return gs.best_Score_,gs.best_params_
