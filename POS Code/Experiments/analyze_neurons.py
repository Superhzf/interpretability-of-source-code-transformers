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


def plot_distribution(fig,ax,top_neurons,model_name):
    distribution = []
    for this_neuron in top_neurons:
        layer = this_neuron//768
        distribution.append(layer)
    data = {}
    # 13 layers
    for this_layer in range(13):
        data[this_layer] = distribution.count(this_layer)
    ax.bar(list(data.keys()), list(data.values()), color ='b',
        width = 0.4)
    ax.set_xlabel(model_name)

folder_name = "distribution_all"
mkdir_if_needed(f"./{folder_name}/")
fig, ((ax1,axNone),(ax2, ax3), (ax4, ax5),(ax6,ax7)) = plt.subplots(4,2,sharex=True,figsize=(10, 9))
fig.text(0.1, 0.5, 'The number of neurons selected', va='center',rotation='vertical')
fig.text(0.5, 0.1, 'Layers', ha='center')
fig.delaxes(axNone)
subplots={'pretrained_BERT':ax1,
          'pretrained_CodeBERT':ax2,'pretrained_GraphCodeBERT':ax3,
          'finetuned_defdet_CodeBERT':ax4,'finetuned_defdet_GraphCodeBERT':ax5,
          'finetuned_clonedet_CodeBERT':ax6,'finetuned_clonedet_GraphCodeBERT':ax7}
for this_model_name in MODEL_NAMES:
    this_regex = re.compile(f'{this_model_name} top neurons\narray\(\[([\S\s]*)\]\)\n{this_model_name} top neurons per class\n',
                            re.MULTILINE)
    this_top_neurons = str2int_top_neurons(this_regex)
    this_ax = subplots[this_model_name]
    plot_distribution(fig,this_ax,this_top_neurons,this_model_name)

plt.savefig(f"./{folder_name}/neuron_dist.png")
