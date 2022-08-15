import os
from run_neurox1 import bert_idx, bert_top_neurons
from run_neurox1 import codebert_idx, codebert_top_neurons
from run_neurox1 import graphcodebert_idx, graphcodebert_top_neurons
from run_neurox1 import bert_class, codebert_class, graphcodebert_class


bert_names = []
for this_neuron in bert_top_neurons:
    for this_idx in bert_idx:
        layer_idx = this_neuron//768
        neuron_idx = this_neuron%768
        this_name = f"bert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
        bert_names.append(this_name)

os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
            svg_stack-main/svg_stack.py result/{bert_names[0]} result/space.svg \
            result/{bert_names[1]} result/space.svg result/{bert_names[2]} \
            result/space.svg result/{bert_names[3]}> result/bert_{bert_class}.svg")


codebert_names = []
for this_neuron in codebert_top_neurons:
    for this_idx in codebert_idx:
        layer_idx = this_neuron//768
        neuron_idx = this_neuron%768
        this_name = f"codebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
        codebert_names.append(this_name)

os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
            svg_stack-main/svg_stack.py result/{codebert_names[0]} result/space.svg \
            result/{codebert_names[1]} result/space.svg result/{codebert_names[2]} \
            result/space.svg result/{codebert_names[3]}> result/codebert_{codebert_class}.svg")


graphcodebert_names = []
for this_neuron in graphcodebert_top_neurons:
    for this_idx in graphcodebert_idx:
        layer_idx = this_neuron//768
        neuron_idx = this_neuron%768
        this_name = f"graphcodebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
        graphcodebert_names.append(this_name)

os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
            svg_stack-main/svg_stack.py result/{graphcodebert_names[0]} result/space.svg \
            result/{graphcodebert_names[1]} result/space.svg result/{graphcodebert_names[2]} \
            result/space.svg result/{graphcodebert_names[3]}> result/graphcodebert_{graphcodebert_class}.svg")
