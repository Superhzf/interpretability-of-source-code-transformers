import os
from run_neurox1 import layer
from run_neurox1 import bert_idx, bert_top_neurons

bert_names = []
for this_neuron in bert_top_neurons:
    for this_idx in bert_idx:
        this_name = f"result/bert_{this_idx-1}_{layer}_{this_neuron-1}.svg"
        bert_names.append(this_name)

os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
            svg_stack-main/svg_stack.py result/{bert_names[0]} result/space.svg \
            result/{bert_names[1]} result/space.svg result/{bert_names[2]} \
            result/space.svg result/{bert_names[3]}> result/bert.svg")
