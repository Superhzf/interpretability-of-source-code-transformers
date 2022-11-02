import os
from run_neurox1 import FOLDER_NAME,MODEL_NAMES,IDX,TOP_NEURONS, CLASSES


def merge_svg(top_neurons,idx,class_name,model_name,folder_name):
    names = []
    for this_neuron in top_neurons:
        for this_idx in idx:
            layer_idx = this_neuron//768
            neuron_idx = this_neuron%768
            this_name = f"{model_name}_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
            names.append(this_name)


    os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
                svg_stack-main/svg_stack.py {folder_name}/{names[0]} {folder_name}/space.svg \
                {folder_name}/{names[1]} {folder_name}/space.svg {folder_name}/{names[2]}> \
                {folder_name}/{model_name}_{class_name}.svg")
    # os.system(f"/work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/bin/python \
    #             svg_stack-main/svg_stack.py {folder_name}/{names[0]} {folder_name}/space.svg \
    #             {folder_name}/{names[1]} {folder_name}/space.svg {folder_name}/{names[2]} \
    #             {folder_name}/space.svg {folder_name}/{names[3]}> {folder_name}/{model_name}_{class_name}.svg")

for this_model in MODEL_NAMES:
    if this_model in ['pretrained_GraphCodeBERT']:
        this_top_neuron = TOP_NEURONS[this_model]
        this_idx = IDX[this_model]
        this_class_name = CLASSES[this_model]
        merge_svg(this_top_neuron,this_idx,this_class_name,this_model,FOLDER_NAME)
        break
