# Interpretability of source code transformers

This folder aims to generate code and associated labels. Make sure to **change
the path to your work space**. The ETA to run the `script` without the validation
part would take less than 10 minutes, the validation part would take ~2 hours.

Only the first 500 Python files of the evaluation dataset in the ETH Python 150 dataset
are used because the evaluation dataset is small and we do not need many Python files
to do the probing task.

## Download the datasets

The datasets can be downloaded from https://www.sri.inf.ethz.ch/py150 under
Version 1.0 Files [190MB].

Then, make a folder and name it `py_150_files`;

Next, unzip the files into the folder;

Finally, inside the `py_150_files` folder, unzip the `data.tar.gz` file.


## Create ml-gpu environment NeuroX_env using the following steps:
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

## Install Neurox from source (pip install version is not updated with Selectivity/Control task code)
```
git clone https://github.com/fdalvi/NeuroX.git  
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install -e .  
```   

## Steps to generate POS dataset from text file containing python code  
We use the python tokenizer to tokenize the code and obtain POS labels. You can
achieve by running `sbatch script`. Basically, it does the folling steps:

1. Merge python files into one single file;

2. Generate codetest.in and codetest.label;

3. Remove observations and labels that have different length;

4. Remove duplicated lines;

5. Validate the result by extracting activations (this is optional).

As a result, `codetest2_unique.in` and `codetest2_unique.label` are ready for
further analysis. Each includes ~32,000 observations.
