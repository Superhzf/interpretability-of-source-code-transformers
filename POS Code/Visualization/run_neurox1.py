"""
Created on Tue Apr 12 14:21:21 2022

@author: sharm
"""
import torch
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.analysis.visualization as vis
import neurox.analysis.corpus as corpus
import os


bert_idx = [382,612,697,734]
bert_top_neurons = [2903]
bert_class = "GREATER"
codebert_idx = [3,4,7,8,9,11]
codebert_top_neurons = [1938]
codebert_class = "KEYWORD"
graphcodebert_idx = [5,28,53,67,82]
graphcodebert_top_neurons = [32]
graphcodebert_class = "EQUAL"


#Extract activations.json files
def extract_activations():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Extract representations from BERT
    transformers_extractor.extract_representations('bert-base-uncased',
        'codetest2_unique.in',
        'bert_activations.json',
        device=device,
        aggregation="average",#last, first
    )

    #Extract representations from CodeBERT
    transformers_extractor.extract_representations('microsoft/codebert-base',
        'codetest2_unique.in',
        'codebert_activations.json',
        device=device,
        aggregation="average", # #last, first
    )

    #Extract representations from GraphCodeBERT
    transformers_extractor.extract_representations('microsoft/graphcodebert-base',
        'codetest2_unique.in',
        'graphcodebert_activations.json',
        device=device,
        aggregation="average",#last, first
    )

    return(load_extracted_activations())


def load_extracted_activations(dev):
    if dev:
        bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
        return bert_activations
    else:
        #Load activations from json files
        bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
        codebert_activations, codebert_num_layers = data_loader.load_activations('codebert_activations.json',13) #num_layers is 13 not 768
        graphcodebert_activations, graphcodebert_num_layers = data_loader.load_activations('graphcodebert_activations.json',13)

        return bert_activations, codebert_activations, graphcodebert_activations


def load_tokens(bert_activations,codebert_activations=None, graphcodebert_activations=None,dev=True):
    if dev:
        bert_tokens = data_loader.load_data('codetest2_unique.in',
                                       'codetest2_unique.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )
        return bert_tokens
    else:
        #Load tokens and sanity checks for parallelism between tokens, labels and activations
        bert_tokens = data_loader.load_data('codetest2_unique.in',
                                       'codetest2_unique.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )

        codebert_tokens = data_loader.load_data('codetest2_unique.in',
                                       'codetest2_unique.label',
                                       codebert_activations,
                                       512 # max_sent_length
                                      )

        graphcodebert_tokens = data_loader.load_data('codetest2_unique.in',
                                       'codetest2_unique.label',
                                       graphcodebert_activations,
                                       512 # max_sent_length
                                      )


        return bert_tokens, codebert_tokens, graphcodebert_tokens



def visualization(bert_tokens, bert_activations,
                  codebert_tokens = None, codebert_activations = None,
                  graphcodebert_tokens = None, graphcodebert_activations = None,
                  dev=True):
    # viz_bert = TransformersVisualizer('bert-base-uncased')
    # viz_codebert = TransformersVisualizer('microsoft/codebert-base')
    # viz_graphcoderbert = TransformersVisualizer('microsoft/graphcodebert-base')

    if dev:
        # starting from 1.
        for this_neuron in bert_top_neurons:
            for this_idx in bert_idx:
                this_svg_bert = vis.visualize_activations(bert_tokens["source"][this_idx-1],
                                                     bert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                name = f"result/bert_{this_idx-1}_{layer}_{this_neuron}.svg"
                this_svg_bert.saveas(name,pretty=True, indent=2)
    else:
        # starting from 1.
        for this_neuron in bert_top_neurons:
            for this_idx in bert_idx:
                this_svg_bert = vis.visualize_activations(bert_tokens["source"][this_idx-1],
                                                     bert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/bert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_bert.saveas(name,pretty=True, indent=2)

        for this_neuron in codebert_top_neurons:
            for this_idx in codebert_idx:
                this_svg_codebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     codebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/codebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_codebert.saveas(name,pretty=True, indent=2)

        for this_neuron in graphcodebert_top_neurons:
            for this_idx in graphcodebert_idx:
                this_svg_graphcodebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     graphcodebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/graphcodebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_graphcodebert.saveas(name,pretty=True, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')
    parser.add_argument("--dev",choices=('True','False'), default='True')
    args = parser.parse_args()
    if args.extract == 'True':
        bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )


 # MINEQUAL
    if args.dev == 'True':
        bert_activations = load_extracted_activations(True)
        bert_tokens =  load_tokens(bert_activations, None, None,True)
        print("Length of bert_activations:",len(bert_activations))
        print("Length of bert_tokens source:",len(bert_tokens["source"]))
        _, num_neurons = bert_activations[0].shape
        for idx in range(len(bert_activations)):
            assert bert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)
        visualization(bert_tokens, bert_activations, None, None, None, None, True)
    else:
        bert_activations, codebert_activations, graphcodebert_activations = load_extracted_activations(False)
        bert_tokens, codebert_tokens, graphcodebert_tokens =  load_tokens(bert_activations, codebert_activations, graphcodebert_activations,False)
        print("Length of bert_activations:",len(bert_activations))
        print("Length of bert_tokens source:",len(bert_tokens["source"]))
        _, num_neurons = bert_activations[0].shape
        for idx in range(len(bert_activations)):
            assert bert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        print("Length of codebert_activations:",len(codebert_activations))
        print("Length of codebert_tokens source:",len(codebert_tokens["source"]))
        _, num_neurons = codebert_activations[0].shape
        for idx in range(len(codebert_activations)):
            assert codebert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        print("Length of graphcodebert_activations:",len(graphcodebert_activations))
        print("Length of graphcodebert_tokens source:",len(graphcodebert_tokens["source"]))
        _, num_neurons = graphcodebert_activations[0].shape
        for idx in range(len(graphcodebert_activations)):
            assert graphcodebert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        visualization(bert_tokens, bert_activations,
                      codebert_tokens,codebert_activations,
                      graphcodebert_tokens,graphcodebert_activations,
                      False)

if __name__ == "__main__":
    main()
