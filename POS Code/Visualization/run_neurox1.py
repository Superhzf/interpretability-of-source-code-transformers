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
bert_idx = [42,114]
bert_top_neurons = [9772]
bert_class = "STRING"
# bert_idx = [3,11]
# bert_top_neurons = [7008]
# bert_class = "KEYWORD"

# codebert_idx = [3,4,14]
# codebert_top_neurons = [6875]
# codebert_class = "NAME"
codebert_idx = [5,59,91,114]
codebert_top_neurons = [2961]
codebert_class = "STRING"

graphcodebert_idx = [53,144,147]
graphcodebert_top_neurons = [3384]
graphcodebert_class = "NUMBER"

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
    activations, num_layers = data_loader.load_activations(activation_file_name,13)
    return activations


def load_tokens(activations):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens = data_loader.load_data('./src_files/codetest2_train_unique.in',
                                   './src_files/codetest2_train_unique.label',
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
        if this_model in ['pretrained_GraphCodeBERT']:
            print(f"Generate svg files for {this_model}")
            this_activation_name = ACTIVATION_NAMES[this_model]
            activations = load_extracted_activations(this_activation_name)
            tokens =  load_tokens(activations)
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
