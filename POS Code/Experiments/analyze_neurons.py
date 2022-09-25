import re
import matplotlib.pyplot as plt
import os
from run_neurox1 import load_extracted_activations
from run_neurox1 import load_tokens
import neurox.analysis.corpus as corpus
from run_neurox1 import MODEL_NAMES

with open('log_all') as f:
    lines = f.read()
f.close()

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def str2int_top_neurons(regex):
    top_neurons = re.findall(regex, lines)[0].replace("\n","")
    top_neurons = top_neurons.split(",")
    top_neurons = [int(this_neuron) for this_neuron in top_neurons]
    return top_neurons


def plot_distribution(top_neurons,model_name):
    distribution = []
    for this_neuron in top_neurons:
        layer = this_neuron//768
        distribution.append(layer)
    data = {}
    # 13 layers
    for this_layer in range(13):
        data[this_layer] = distribution.count(this_layer)
    fig = plt.figure(figsize = (10, 5))
    plt.bar(list(data.keys()), list(data.values()), color ='b',
        width = 0.4)
    plt.savefig(f"./{folder_name}/{model_name}_neuron_dist.png")

folder_name = "distribution_all"
mkdir_if_needed(f"./{folder_name}/")
for this_model_name in MODEL_NAMES:
    this_regex = re.compile(f'{this_model_name} top neurons\narray\(\[([\S\s]*)\]\)\n{this_model_name} top neurons per class\n',
                            re.MULTILINE)
    this_top_neurons = str2int_top_neurons(this_regex)
    plot_distribution(this_top_neurons,this_model_name)
