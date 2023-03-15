import math
import numpy as np
import torch
import torch.nn as nn
import skorch
import os
import glob as glob
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm as progressbar
from . import utils
from . import metrics
from ray import tune
from ray.air import session

## Regularizers
def l1_penalty(var):
    return torch.abs(var).sum()


def l2_penalty(var):
    return torch.sqrt(torch.pow(var, 2).sum())


## Train helpers
def batch_generator(X, y, batch_size=32):
    start_idx = 0
    while start_idx < X.shape[0]:
        yield X[start_idx : start_idx + batch_size], y[
            start_idx : start_idx + batch_size
        ]
        start_idx = start_idx + batch_size


#pass training data    
def train_model(
    X_tensor,
    y_tensor,
    model,
    optimizer,
    epoch,
    num_epochs,
    use_gpu,
    model_type,
    criterion,
    lambda_l1=0.0001,
    lambda_l2=0.0001,
    batch_size=32,
    learning_rate=0.001,
    metric='accuracy'
):
    train_running_correct = 0
    num_tokens = 0
    avg_loss = 0
    for inputs, labels in progressbar(
        batch_generator(X_tensor, y_tensor, batch_size=batch_size),
        desc="epoch [%d/%d]" % (epoch + 1, num_epochs),
    ):
        num_tokens += inputs.shape[0]
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)

        # Forward + Optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        if model_type == "regression":
            outputs = outputs.squeeze()
        weights = list(model.parameters())[0]

        loss = (
            criterion(outputs, labels)
            + lambda_l1 * l1_penalty(weights)
            + lambda_l2 * l2_penalty(weights)
        )

        _, predicted = torch.max(outputs.data, 1)
        train_running_correct += (predicted == labels).sum().item()
	#Backpropagation___________________________
        loss.backward()
        optimizer.step()

        avg_loss += loss.item()
   # session.report({"loss": avg_loss/num_tokens})  # This sends the score to Tune.

    #y_pred = np.array(y_pred)

   # result = metrics.compute_score(y_pred, y, metric)

    epoch_acc = 100. * (train_running_correct / len(X_tensor))
    print(
	"Epoch: [%d/%d], Loss: %.4f"
	% (epoch + 1, num_epochs, avg_loss / num_tokens)
    )
    epoch_loss = avg_loss/num_tokens 

    return avg_loss, epoch_acc      

#pass validation data
def validate_model(
    X_tensor,
    y_tensor,
    model,
    optimizer,
    epoch,
    num_epochs,
    use_gpu,
    model_type,
    criterion,
    lambda_l1=0.001,
    lambda_l2=0.001,
    batch_size=32,
    learning_rate=0.001,
    metric='accuracy'
):
    valid_running_correct = 0  
    num_tokens = 0
    avg_loss = 0
    for inputs, labels in progressbar(
        batch_generator(X_tensor, y_tensor, batch_size=batch_size),
        desc="epoch [%d/%d]" % (epoch + 1, num_epochs),
    ):
        num_tokens += inputs.shape[0]
        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
        inputs = Variable(inputs)
        labels = Variable(labels)

        # Forward + Optimize
        optimizer.zero_grad()
        outputs = model(inputs)
        if model_type == "regression":
            outputs = outputs.squeeze()
        weights = list(model.parameters())[0]

        loss = (
            criterion(outputs, labels)
            + lambda_l1 * l1_penalty(weights)
            + lambda_l2 * l2_penalty(weights)
        )
        
        _, predicted = torch.max(outputs.data, 1)

        valid_running_correct += (predicted == labels).sum().item()

        #loss.backward()
        optimizer.step()

        avg_loss += loss.item()
     
    
    #y_pred = np.array(y_pred)
    #result = metrics.compute_score(y_pred, y, metric)
    epoch_acc = 100. * (valid_running_correct / len(X_tensor))
   # print("Score (%s) of the model: %0.2f" % (metric, result))

    print("Epoch: [%d/%d], Loss: %.4f"
        % (epoch + 1, num_epochs, avg_loss / num_tokens)
    )
    epoch_loss = avg_loss/num_tokens 
    with tune.checkpoint_dir(epoch) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((model.state_dict(), optimizer.state_dict()), path)

    tune.report(loss=epoch_loss, accuracy=epoch_acc)
    
    return epoch_loss, epoch_acc


