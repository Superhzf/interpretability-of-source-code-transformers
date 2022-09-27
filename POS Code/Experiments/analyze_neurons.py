import re
import matplotlib.pyplot as plt
import os
from run_neurox1 import MODEL_NAMES
import numpy as np

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


def str2top_neurons_per_class(regex):
    top_neurons_per_class = re.findall(this_regex, lines)[0].replace("\n","")
    top_neurons_per_class = top_neurons_per_class.replace('array','np.array')
    top_neurons_per_class = eval(top_neurons_per_class)
    return top_neurons_per_class


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
    plt.ylim(0, 70)
    plt.savefig(f"./{folder_name}/{model_name}_neuron_dist.png")
    plt.close()

def plot_classVSneurons(height,pos,bar_width,color,model_name):
    plt.bar(pos, height, color =color, width = bar_width,
        edgecolor ='grey', label =model_name)



folder_name = "distribution_all"
mkdir_if_needed(f"./{folder_name}/")


for this_model_name in MODEL_NAMES:
    this_regex = re.compile(f'{this_model_name} top neurons\narray\(\[([\S\s]*)\]\)\n{this_model_name} top neurons per class\n',
                            re.MULTILINE)
    this_top_neurons = str2int_top_neurons(this_regex)
    plot_distribution(this_top_neurons,this_model_name)


features_selected = {'pretrained_BERT':475,
                     'pretrained_CodeBERT':425,'pretrained_GraphCodeBERT':411,
                     'finetuned_defdet_CodeBERT':374,'finetuned_defdet_GraphCodeBERT':389,
                     'finetuned_clonedet_CodeBERT':416,'finetuned_clonedet_GraphCodeBERT':410}

classes_interested = ['INDENT','NAME',"KEYWORD","NOTEQUAL","GREATER"]

colors = ["silver", "lightgreen", "lightcyan", "lightsteelblue", "lightyellow", "lightblue", "lightpink"]

bar_width = 0.1
fig = plt.subplots(figsize =(20, 8))

for count,this_model_name in enumerate(MODEL_NAMES):
    this_features_selected = features_selected[this_model_name]
    this_regex = re.compile(f'{this_model_name} top neurons per class\n([\S\s]*)\nThe shape of selected features \(260064, {this_features_selected}\)',
                        re.MULTILINE)
    this_top_neurons_per_class = str2top_neurons_per_class(this_regex)
    this_height = []
    for this_class in classes_interested:
        this_height.append(len(this_top_neurons_per_class[this_class]))
    if count == 0:
        this_pos = np.arange(len(classes_interested))
    else:
        this_pos = [x + bar_width for x in this_pos]
    this_color = colors[count]
    plot_classVSneurons(this_height,this_pos,bar_width,this_color,this_model_name)

plt.xticks([r + 3*bar_width for r in range(len(classes_interested))],
        classes_interested)
plt.legend()
plt.savefig(f"./{folder_name}/neurons_per_class.png")
