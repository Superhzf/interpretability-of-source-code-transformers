import argparse
import json
import os
import sys

import numpy as np
import psutil
import torch.optim as optim
import os
import math
import numpy as np
import torch
import torch.nn as nn

from imblearn.under_sampling import RandomUnderSampler
from torch.autograd import Variable

from collections import defaultdict
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from tqdm import tqdm as progressbar
# Load aux classifier
if 'AUX_CLASSIFIER_PATH' in os.environ:
    sys.path.append(os.environ['AUX_CLASSIFIER_PATH'])
from aux_classifier import data_loader
from aux_classifier import utils
from aux_classifier import ranking
from aux_classifier import metrics
from aux_classifier import grid_search  

#config
# Training parameters.
EPOCHS = 50
# For ASHA scheduler in Ray Tune.
MAX_NUM_EPOCHS = 50
GRACE_PERIOD = 1
# For search run (Ray Tune settings).
CPU = 8
GPU = 0
# Number of random search experiments to run.
NUM_SAMPLES = 2

CONFIGURATION_OPTIONS = {
    # Data directories
    "base_dir": {
        "example": "/some/path/to/files",
        "type": "string",
        "description": "Path to the directory containing data files",
    },
    "train_source": {
        "example": "training_data.word",
        "type": "string",
        "description": "Training tokens file",
    },
    "train_labels": {
        "example": "training_data.labels",
        "type": "string",
        "description": "Training labels file",
    },
    "train_activations": {
        "example": "training_data.hdf5",
        "type": "string",
        "description": "Training activations file",
    },
    "test_source": {
        "example": "test_data.word",
        "type": "string",
        "description": "Test tokens file",
    },
    "test_labels": {
        "example": "test_data.labels",
        "type": "string",
        "description": "Test labels file",
    },
    "test_activations": {
        "example": "test_data.hdf5",
        "type": "string",
        "description": "Test activations file",
    },
    "task_specific_tag": {
        "example": "N",
        "type": "string",
        "description": "Task specific label for unknown labels in test data",
    },
    "output_directory": {
        "example": "path/to/output/directory",
        "type": "string",
        "description": "Path where experiment results should be saved",
    },
    # Experiment variables
    "max_sent_l": {
        "example": 1000,
        "type": "integer",
        "description": "Maximum length of the input sentence",
    },
    "is_brnn": {
        "example": False,
        "type": "boolean",
        "description": "Whether the trained model is a bidirectional model",
    },
    "num_neurons_per_layer": {
        "example": 1024,
        "type": "integer",
        "description": """Number of neurons in every layer of the trained model.
                        Example: BERT: 768, ELMO: 1024""",
    },
    # Ranking variables
    "limit_instances": {
        "example": 40000,
        "type": "integer",
        "description": """Number of instances to limit the training to. Data is sampled
                        proportionally depending on overall class distribution""",
    },
    "clustering_thresholds": {
        "example": [-1, 0.3],
        "type": "list",
        "description": "Thresholds for clustering (Set to -1 for no clustering)",
    },
    "ranking_type": {
        "example": "multiclass",
        "type": "string",
        "description": "Type of ranking to try. Acceptable values: multiclass, binary, multiclasscv, binarycv",
    },
    # Optimization variables
    "num_epochs": {"example": 10, "type": "integer", "description": "Number of epochs"},
    "batch_size": {"example": 128, "type": "integer", "description": "Batch Size"},
    "lambda_l1": {
        "example": 0.00001,
        "type": "float",
        "description": "Regularization L1 parameter",
    },
    "lambda_l2": {
        "example": 0.00001,
        "type": "float",
        "description": "Regularization L2 parameter",
    },
    "model_type": {
        "example": "classification",
        "type": "string",
        "description": "classification/regression",
    },
    "metric": {
        "example": "accuracy",
        "type": "string",
        "description": "accuracy/f1/accuracy_and_f1/pearson/spearman/pearson_and_spearman/matthews_corrcoef",
    },
    # Selection variables
    "performance_deltas": {
        "example": [(3, 1), (2, 1), (1, 1)],
        "type": "list of tuples",
        "description": "Percentage of relative reduction in accuracy allowed while selecting number of layers and choosing minimal neuron set",
    },
}


def memory_usage_psutil():
    # return the memory usage in MB
    process = psutil.Process(os.getpid())
    mem = process.memory_info()[0] / float(2 ** 20)
    return mem


def print_sample_configuration():
    props = list(CONFIGURATION_OPTIONS.keys())
    print("{")
    for p in props:
        print(
            "\t%s:%s, # %s"
            % (
                p,
                str(CONFIGURATION_OPTIONS[p]["example"]),
                CONFIGURATION_OPTIONS[p]["description"],
            )
        )
    print("}")


def load_configuration(config_path):
    with open(config_path, "r") as fp:
        config = json.load(fp)
    return config


def is_config_valid(config):  #model_config
    provided_props = set(config.keys())
    required_props = set(CONFIGURATION_OPTIONS.keys())

    missing_props = list(required_props - provided_props)
    extra_props = list(provided_props - required_props)

    if len(missing_props) > 0:
        print("Some props are missing:")
        print(missing_props)
        return False

    if len(extra_props) > 0:
        print("You included some extra props:")
        print(extra_props)
        return False

    return True


def train_and_validate(config,checkpoint_dir=None): #this is param_config

    X_train, y_train,X_dev,y_dev,X_test,y_test,model_type,learning_rate,num_epochs=get_data()    

    print("Training %s model" % (model_type))
    # Check if we can use GPU's for training
    use_gpu = torch.cuda.is_available()

    #for evaluation
    y_pred = []
    def source_generator():
        for s in source_tokens:
            for t in s:
                yield t
    src_words = source_generator()
   # if return_predictions:
    predictions = []
   # else:
    #    src_word = -1

    print("Creating model...")
    if model_type == "classification":
        num_classes = len(set(y_train))
        assert (
            num_classes > 1
        ), "Classification problem must have more than one target class"
    else:
        num_classes = 1
    print("Number of training instances:", X_train.shape[0])
    if model_type == "classification":
        print("Number of classes:", num_classes)

    model = utils.LinearNet(X_train.shape[1], num_classes)

    if use_gpu:
        model = model.cuda()

    if model_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif model_type == "regression":
        criterion = nn.MSELoss()
    else:
        assert (
            model_type == "classification" or model_type == "regression"
        ), "Invalid model type"

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.from_numpy(y_train)

    for epoch in range(num_epochs):
    #call train and valid functions and return
        train_epoch_loss, train_epoch_acc = grid_search.train_model(X_tensor,y_tensor,model,optimizer,epoch,num_epochs,use_gpu,model_type,criterion,config['lambda_l1'],config['lambda_l2'], batch_size=config['batch_size'], learning_rate=config['lr'],metric='f1')
        valid_epoch_loss, valid_epoch_acc  = grid_search.validate_model(X_tensor,y_tensor,model,optimizer,epoch,num_epochs,use_gpu,model_type,criterion,config['lambda_l1'],config['lambda_l2'], batch_size=config['batch_size'], learning_rate=config['lr'],metric='f1')


#        with tune.checkpoint_dir(epoch) as checkpoint_dir:
 #           path = os.path.join(checkpoint_dir, 'checkpoint')
  #          torch.save((model.state_dict(), optimizer.state_dict()), path)
       # tune.report(
       #     loss=valid_epoch_loss, accuracy=valid_epoch_acc
       # )



def run_search():
    os.environ["TUNE_MAX_PENDING_TRIALS_PG"] = "1"
    # Define the parameter search configuration.
    param_config = {
        "lambda_l1":
             tune.grid_search([1e-8, 1e-7,1e-6, 1e-5,1e-4, 1e-3, 1e-2, 1e-1,0]),
#            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "lambda_l2":
             tune.grid_search([1e-8, 1e-7, 1e-6, 1e-5,1e-4 1e-3, 1e-2, 1e-1,0]),
#            tune.sample_from(lambda _: 2 ** np.random.randint(4, 8)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([ 32, 56,128])
    }
    # Schduler to stop bad performing trails.
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=MAX_NUM_EPOCHS,
        grace_period=GRACE_PERIOD,
        reduction_factor=2)
    # Reporter to show on command line/output window
    reporter = CLIReporter(
            metric_columns=["loss", "accuracy", "training_iteration"])

    # Start run/search
    result = tune.run(
        train_and_validate,
        resources_per_trial={"cpu": CPU, "gpu": GPU},
        max_concurrent_trials=4,
        config=param_config,
        num_samples=NUM_SAMPLES, #for random search results
        scheduler=scheduler,
        local_dir='../outputs/raytune_result',
        keep_checkpoints_num=1,
        checkpoint_score_attr='min-loss',
        progress_reporter=reporter
    )
    # Extract the best trial run from the search.
    best_trial = result.get_best_trial(
            'loss', 'min' 
    )
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation acc: {best_trial.last_result['accuracy']}")

#    best_trial = result.get_best_trial(
 #           'accuracy', 'max'
#    )
 #   print(f"Best trial config: {best_trial.param_config}")
  #  print(f"Best trial final validation loss: {best_trial.last_result['validation_loss']}")
   # print(f"Best trial final validation acc: {best_trial.last_result['accuracy']}")
     

def get_data():  #pass model_config here 
    config = model_config
    all_results = {}    
    train_source_path = os.path.join(config["base_dir"], config["train_source"])
    train_labels_path = os.path.join(config["base_dir"], config["train_labels"])
    train_activations_path = os.path.join(
        config["base_dir"], config["train_activations"]
    )

    # dev_source_path = os.path.join(config["base_dir"], config["dev_source"])
    # dev_labels_path = os.path.join(config["base_dir"], config["dev_labels"])
    # dev_activations_path = os.path.join(config["base_dir"], config["dev_activations"])

    test_source_path = os.path.join(config["base_dir"], config["test_source"])
    test_labels_path = os.path.join(config["base_dir"], config["test_labels"])
    test_activations_path = os.path.join(config["base_dir"], config["test_activations"])

    print("*********************** LOADING ACTIVATIONS ***********************")
    print("[MEMORY] Before activation loading: %0.2f" % (memory_usage_psutil()))
    train_activations, num_layers = data_loader.load_activations(
        train_activations_path,
        config["num_neurons_per_layer"],
        is_brnn=config["is_brnn"],
    )
    # dev_activations, _ = data_loader.load_activations(
    #     dev_activations_path, config["num_neurons_per_layer"], is_brnn=config["is_brnn"]
    # )
    test_activations, _ = data_loader.load_activations(
        test_activations_path,
        config["num_neurons_per_layer"],
        is_brnn=config["is_brnn"],
    )
    print("Number of train sentences: %d" % (len(train_activations)))
    print("Number of test sentences: %d" % (len(test_activations)))

    print("************************* LOADING TOKENS **************************")
    train_tokens = data_loader.load_sentence_data(
        train_source_path, train_labels_path, train_activations
    )
    # dev_tokens = data_loader.load_sentence_data(
    #     dev_source_path, dev_labels_path, dev_activations
    # )
    test_tokens = data_loader.load_sentence_data(
        test_source_path, test_labels_path, test_activations
    )
    print("[MEMORY] After token loading: %0.2f" % (memory_usage_psutil()))

    print("************************ CREATING TENSORS *************************")
    print("Train:")
    X_full, y_full, mappings = utils.create_tensors(
        train_tokens,
        train_activations,
        config["task_specific_tag"],
        model_type=config["model_type"],
    )
    
    # X_dev, y_dev, mappings = utils.create_tensors(
    #     dev_tokens,
    #     dev_activations,
    #     config["task_specific_tag"],
    #     mappings,
    #     model_type=config["model_type"],
    # )
    np.random.seed(7361)
    all_idx = np.arange(X_full.shape[0])
    np.random.shuffle(all_idx)
    num_train_examples = int(X_full.shape[0] * 0.90) # Select 90% for training
    train_idx = all_idx[:num_train_examples]
    dev_idx = all_idx[num_train_examples:]

    all_results['data_split'] = {
        'all_idx': [int(zz) for zz in all_idx],
        'train_idx': [int(zz) for zz in train_idx],
        'dev_idx': [int(zz) for zz in dev_idx]
    }
    
    X = X_full[train_idx, :]
    y = y_full[train_idx]

    X_dev = X_full[dev_idx, :]
    y_dev = y_full[dev_idx]

    print(X.shape, y.shape)
    print("Dev:")
    print(X_dev.shape, y_dev.shape)
    print("Test:")
    X_test, y_test, mappings = utils.create_tensors(
        test_tokens,
        test_activations,
        config["task_specific_tag"],
        mappings,
        model_type=config["model_type"],
    )
    print(X_test.shape, y_test.shape)
    print("[MEMORY] After tensor creation: %0.2f" % (memory_usage_psutil()))

    print("************************* FREEING MEMORY **************************")
    import gc

    train_tokens["source"] = None
    train_tokens["target"] = None
    train_tokens = None

    for idx, _ in enumerate(train_activations):
        train_activations[idx] = None
    train_activations = None

    # dev_tokens["source"] = None
    # dev_tokens["target"] = None
    # dev_tokens = None

    # for idx, _ in enumerate(dev_activations):
    #     dev_activations[idx] = None
    # dev_activations = None

    test_tokens["source"] = None
    test_tokens["target"] = None
    test_tokens = None
    for idx, _ in enumerate(test_activations):
        test_activations[idx] = None
    test_activations = None

    gc.collect()
    print("[MEMORY] After cleanup: %0.2f" % (memory_usage_psutil()))

    model_type=config["model_type"]
    learning_rate = 1e-2
    num_epochs= config["num_epochs"]
    return X,y,X_dev,y_dev,X_test,y_test,model_type,learning_rate,num_epochs


def main():
    print("[MEMORY] Begin: %0.2f" % (memory_usage_psutil()))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", dest="config_path", help="Path to configuration file"
    )

    args = parser.parse_args()
    #load model_config here
    if not args.config_path:
        print(
            "Please provide an experiment configuration file, a sample is shown below:"
        )
        print_sample_configuration()
        sys.exit()

    global model_config
    model_config = load_configuration(args.config_path)

    if not is_config_valid(model_config):
        sys.exit()
   
    print("***************************** RUNNING GRIDSEARCH *****************************")
    run_search()



if  __name__ == "__main__":
    main()
