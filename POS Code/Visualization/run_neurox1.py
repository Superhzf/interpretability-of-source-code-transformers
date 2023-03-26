import torch
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.analysis.visualization as vis
import neurox.analysis.corpus as corpus
import os


MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',
               'finetuned_defdet_CodeBERT','finetuned_defdet_GraphCodeBERT',
               'finetuned_clonedet_CodeBERT','finetuned_clonedet_GraphCodeBERT']
ACTIVATION_NAMES = {'pretrained_BERT':'bert_activations_train.json',
                    'pretrained_CodeBERT':'codebert_activations_train.json',
                    'pretrained_GraphCodeBERT':'graphcodebert_activations_train.json',
                    'finetuned_defdet_CodeBERT':'codebert_defdet_activations_train.json',
                    'finetuned_defdet_GraphCodeBERT':'graphcodebert_defdet_activations_train.json',
                    'finetuned_clonedet_CodeBERT':'codebert_clonedet_activations1_train.json',
                    'finetuned_clonedet_GraphCodeBERT':'graphcodebert_clonedet_activations1_train.json'}
# This set of idx is for pretrained, finetuned defdet, and finetuned clonedet models
bert_idx = [3488,10833,21767,21782,23331]
bert_top_neurons = [4194]
bert_class = "MIXTURE_KW_NUMBER_NAME"

# bert_idx = [461,543,545,381,384,395,254,270,275,41,695,812,798]
# bert_top_neurons = [4097]
# bert_class = "NAME"


# bert_idx = [4157,10866,17523,17526]
# bert_top_neurons = [8322]
# bert_class = "STRING"

# bert_idx = [179,251,707,241,537,11,301]
# bert_top_neurons = [6295]
# bert_class = "KEYWORD"

# bert_idx = [115,3621,1082,798]
# bert_top_neurons = [7]
# bert_class = "MIXTURE"



codebert_idx=[]
graphcodebert_idx=[]
codebert_top_neurons=[]
graphcodebert_top_neurons=[]
codebert_class =[]
graphcodebert_class=[]


IDX = {"pretrained_BERT":bert_idx,
       "pretrained_CodeBERT":codebert_idx,"pretrained_GraphCodeBERT":graphcodebert_idx,
       "finetuned_defdet_CodeBERT":codebert_idx,"finetuned_defdet_GraphCodeBERT":graphcodebert_idx,
       "finetuned_clonedet_CodeBERT":codebert_idx,"finetuned_clonedet_GraphCodeBERT":graphcodebert_idx}
TOP_NEURONS = {"pretrained_BERT":bert_top_neurons,
               "pretrained_CodeBERT":codebert_top_neurons,"pretrained_GraphCodeBERT":graphcodebert_top_neurons,
               "finetuned_defdet_CodeBERT":codebert_top_neurons,"finetuned_defdet_GraphCodeBERT":graphcodebert_top_neurons,
               "finetuned_clonedet_CodeBERT":codebert_top_neurons,"finetuned_clonedet_GraphCodeBERT":graphcodebert_top_neurons}
CLASSES = {"pretrained_BERT":bert_class,
           "pretrained_CodeBERT":codebert_class,"pretrained_GraphCodeBERT":graphcodebert_class,
           "finetuned_defdet_CodeBERT":codebert_class,"finetuned_defdet_GraphCodeBERT":graphcodebert_class,
           "finetuned_clonedet_CodeBERT":codebert_class,"finetuned_clonedet_GraphCodeBERT":graphcodebert_class}

FOLDER_NAME ="result_all"

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def load_extracted_activations(activation_file_name):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(f"../Experiments/activations/{activation_file_name}")
    return activations


def load_tokens(activations):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens = data_loader.load_data('../Experiments/src_files/codetest2_train_unique.in',
                                   '../Experiments/src_files/codetest2_train_unique.label',
                                   activations,
                                   512 # max_sent_length
                                  )
    return tokens


def visualization(tokens, activations,top_neurons,idx,model_name):
    for this_neuron in top_neurons:
        for this_idx in idx:
            this_svg_bert = vis.visualize_activations(tokens["source"][this_idx-1],
                                                 activations[this_idx-1][:, this_neuron],
                                                 filter_fn="top_tokens")
            layer_idx = this_neuron//768
            neuron_idx = this_neuron%768
            name = f"{FOLDER_NAME}/{model_name}_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
            this_svg_bert.saveas(name,pretty=True, indent=2)


def main():
    mkdir_if_needed(f"./{FOLDER_NAME}/")

    for this_model in MODEL_NAMES:
        if this_model in ['pretrained_BERT']:
            print(f"Generate svg files for {this_model}")
            this_activation_name = ACTIVATION_NAMES[this_model]
            activations = load_extracted_activations(this_activation_name)
            tokens, _ =  load_tokens(activations)
            print(f"Length of {this_model} activations:",len(activations))
            print(f"Length of {this_model} tokens source:",len(tokens["source"]))
            _, num_neurons = activations[0].shape
            for idx in range(len(activations)):
                assert activations[idx].shape[1] == num_neurons
            print(f"The number of neurons for each token in {this_model}:",num_neurons)
            this_idx = IDX[this_model]
            this_top_neurons = TOP_NEURONS[this_model]
            visualization(tokens, activations,this_top_neurons,this_idx,this_model)
            print("-----------------------------------------------------------------")
            break

if __name__ == "__main__":
    main()
