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
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',]
ACTIVATION_NAMES = {'pretrained_BERT':'bert_activations_train.json',
                    'pretrained_CodeBERT':'codebert_activations_train.json',
                    'pretrained_GraphCodeBERT':'graphcodebert_activations_train.json',}

FOLDER_NAME ="result_all"

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def load_extracted_activations(activation_file_name,activation_folder):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(f"../Experiments/{activation_folder}/{activation_file_name}")
    return activations



def cka(activation1,activation2,model_name1,model_name2):
    for this_sample1 in activation1:
        print(f"The shape of this sample:{this_sample1.shape}")
        exit(0)

    


def main():
    mkdir_if_needed(f"./{FOLDER_NAME}/")
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='python')
    args = parser.parse_args()
    language = args.language
    if language == 'python':
        activation_folder = "activations"
        src_folder = "src_files"
    elif language == 'java':
        activation_folder = "activations_java"
        src_folder = "src_java"

    for this_model in MODEL_NAMES:
        if this_model in ['pretrained_CodeBERT']:
            print(f"Generate svg files for {this_model}")
            this_activation_name = ACTIVATION_NAMES[this_model]
            activations = load_extracted_activations(this_activation_name,activation_folder)
            print(f"Length of {this_model} activations:",len(activations))
            _, num_neurons = activations[0].shape
            for idx in range(len(activations)):
                assert activations[idx].shape[1] == num_neurons
            print(f"The number of neurons for each token in {this_model}:",num_neurons)
            cka(activations,activations,model_name1=this_model,model_name2=this_model)
            print("-----------------------------------------------------------------")
            break

if __name__ == "__main__":
    main()
