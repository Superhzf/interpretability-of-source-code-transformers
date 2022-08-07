"""
Created on Tue Apr 12 14:21:21 2022

@author: sharm
"""
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
# from neurox.analysis.visualization import TransformersVisualizer
import neurox.analysis.visualization as viz




#Extract activations.json files
def extract_activations():
    #Extract representations from BERT
    transformers_extractor.extract_representations('bert-base-uncased',
        'codetest2.in',
        'bert_activations.json',
        aggregation="average" #last, first
    )

    #Extract representations from CodeBERT
    transformers_extractor.extract_representations('microsoft/codebert-base',
        'codetest2.in',
        'codebert_activations.json',
        aggregation="average" #last, first
    )

    #Extract representations from GraphCodeBERT
    transformers_extractor.extract_representations('microsoft/graphcodebert-base',
        'codetest2.in',
        'graphcodebert_activations.json',
        aggregation="average" #last, first
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
        bert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )
        return bert_tokens
    else:
        #Load tokens and sanity checks for parallelism between tokens, labels and activations
        bert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )

        codebert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       codebert_activations,
                                       512 # max_sent_length
                                      )

        graphcodebert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       graphcodebert_activations,
                                       512 # max_sent_length
                                      )


        return bert_tokens, codebert_tokens, graphcodebert_tokens



def visualization(bert_tokens, bert_activations,
                  codebert_tokens=None, codebert_activations=None,
                  graphcodebert_tokens=None,graphcodebert_activations=None,
                  dev=True):
    if dev:
        # viz = TransformersVisualizer('bert-base-uncased')
        for s_idx in range(len(bert_tokens["source"])):
            layer=0
            neuron=5
            this_svg=viz.visualize_activations(bert_tokens['source'][s_idx],
                                               bert_activations[s_idx][layer,neuron],
                                               filter_fn="top_tokens")
            # this_svg=viz(bert_tokens["source"][s_idx], layer, neuron, filter_fn="top_tokens")
            image_name = f"bert_{s_idx}.svg"
            this_svg.save(pretty=True, indent=2)
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')
    parser.add_argument("--dev",choices=('True','False'), default='True')
    args = parser.parse_args()
    if args.extract == 'True':
        bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )

    if args.dev:
        bert_activations = load_extracted_activations(True)
        bert_tokens =  load_tokens(bert_activations,None,None,True)
        visualization(bert_tokens, bert_activations, None, None, None, None, True)
    else:
        bert_activations, codebert_activations, graphcodebert_activations = load_extracted_activations(False)
        bert_tokens, codebert_tokens, graphcodebert_tokens =  load_tokens(bert_activations, codebert_activations, graphcodebert_activations,False)
        visualization(bert_tokens, bert_activations,
                      codebert_tokens, codebert_activations,
                      graphcodebert_tokens,graphcodebert_activations,
                      False)


if __name__ == "__main__":
    main()
