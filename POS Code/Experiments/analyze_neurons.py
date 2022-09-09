import re
import matplotlib.pyplot as plt
import os
from run_neurox1 import load_extracted_activations
from run_neurox1 import load_tokens
import neurox.analysis.corpus as corpus

with open('log') as f:
    lines = f.read()
f.close()

regex_bert_top_neurons = re.compile('Bert top neurons\narray\(\[([\S\s]*)\]\)\nBert top neurons per class\n',
                                    re.MULTILINE)
regex_codebert_top_neurons = re.compile('CodeBert top neurons\narray\(\[([\S\s]*)\]\)\nCodeBert top neurons per class\n',
                                    re.MULTILINE)
regex_graphcodebert_top_neurons = re.compile('GraphCodeBert top neurons\narray\(\[([\S\s]*)\]\)\nGraphCodeBert top neurons per class\n',
                                    re.MULTILINE)


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

    plt.xlabel("Layers")
    plt.ylabel("The number of neurons selected")
    plt.title(f"{model_name}:neuron distribution across layers")
    plt.savefig(f"./distribution/{model_name}_neuron_dist.png")


mkdir_if_needed("./distribution/")
model_names = ["BERT","CODEBERT","GRAPHCODEBERT"]
regex_list = [regex_bert_top_neurons,regex_codebert_top_neurons,regex_graphcodebert_top_neurons]
for this_regex, this_model_name in zip(regex_list,model_names):
    this_top_neurons = str2int_top_neurons(this_regex)
    plot_distribution(this_top_neurons,this_model_name)
