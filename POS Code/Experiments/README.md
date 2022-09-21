# Interpretability of source code transformers

## Create ml-gpu environment using the following steps
```
srun --time=00:10:00 --nodes=1 --cpus-per-task=4 --partition=gpu --gres=gpu:1 --pty /usr/bin/bash  
module load ml-gpu  
mkdir NeuroX_env  
ml-gpu python -m venv <path to environment>/NeuroX_env  
cd NeuroX_env
```
## Install PyTorch

The default Neurox installation is not compatible with the most advanced GPU on
pronto. Therefore, we have to install PyTorch first. The required PyTorch version
is 1.8.1
```
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install Neurox from source (pip install version is not updated with Control task code)
```
git clone https://github.com/fdalvi/NeuroX.git  
cd NeuroX
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install -e .  
```
## Run experiments
Update script with correct paths to your environment.  
Set --extract=False if activation files have already been generated.  
Location of activation.json files:
/work/LAS/jannesar-lab/arushi/Interpretability/interpretability-of-source-code-transformers/POS\ Code/Experiments  
```
cd interpretability-of-source-code-transformers/POS\ Code/Experiments
sbatch script
```

## Experiment setups:

1. Having the code and labels, plug into the models and, activations return;

2. With the activations and labels, the dataset for probing task is created;

3. Truncate the dataset such that only the 41 most frequent classes are kept, the minimum number of occurrences is 10;

4. Split the dataset into training (0.8) and testing (0.2), the number of training instances is 260064;

5. Calculate the mean and standard deviation based on the training set and normalize the training and testing set using the mean and standard deviation;

6. Building the probing logistic regression model;

7. Building the control task model;

8. Once having the weights of the probing logistic regression model, get the important neurons, and draw the distribution;

9. Based on the important neurons, get the top words of each neurons, and generate the table to show those words;

10. Finally, based on the important neurons for different classes, select lines of code that include the classes that we are interested in, and highlight the tokens that have a large absolute activation values.
