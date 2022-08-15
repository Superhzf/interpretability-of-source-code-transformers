# Interpretability of source code transformers
This folder aims to generate SVG files indicating how neurons activate different
classes.

The generated SVG files for the paper are in the `result` folder. They are `bert_classname.svg`,
`codebert_classname.svg`, `graphcodebert_classname.svg`. Please keep reading if
you want to know more details.


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

## svg_stack package
After the SVG files for an observation of a certain neuron in a certain layer are
generated, we want to merge those single SVG files into one single SVG file for
one model. We can achieve that with the help of `svg_stack` package.

The `svg_stack` package is readily available in the folder, you don't have to
download it again or install it. Credits go to https://github.com/astraw/svg_stack.

All you have to do is installing the required package [lxml](https://lxml.de/installation.html)
by

```
ml-gpu <path to environment>/NeuroX_env/bin/pip3 install lxml==4.6.3
```


## Generate single SVG files
Update `script_main.sh` with correct paths to your environment and parameters:

Set `--extract=False` if activation files have already been generated.

Set `--dev=True` if it is in the development mode. Running in development mode is fast by only running BERT model

In this folder, run:
```
sbatch script_main.sh
```

Once done, you will find model_obs_layer_neuron.svg files in the `result` folder.
For example `bert_1_2_3.svg` means the second observation on BERT model, 3rd
layer, and 4th neuron.


## Merge single SVG files into one for one model
Update `script_generate_svg.sh` with correct paths to your environment. No parameters
needed.

In this folder, run:
```
sbatch script_generate_svg.sh
```

Then you will see `bert_classname.svg`, `codebert_classname.svg`, and
`graphcodebert_classname.svg` in the `result` folder.

As you can see, there are both red and blue colors in the graphs. Per the [NeuroX](https://neurox.qcri.org/docs/neurox.analysis.html#module-neurox.analysis.visualization) package, blue means the activation value is large
and red means the activation is small. I don't know how to explain why the
activation for a class in the model is larger (small) not small (large).


## Regarding the `get_top_words`
I wrote the code to generate top words without being notified that the code is readily
available in the `../Experiment` folder. I believe the code should be eliminated from here
once top words are found in that folder to keep the whole project clean and concise.
I will keep it for now till getting top words is done.
