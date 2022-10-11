# this script aims to explore 
# 1. whether the same token at the same position will have the same activation value after the first layer

import argparse
from run_neurox1 import load_extracted_activations
import neurox.data.extraction.transformers_extractor as transformers_extractor

def extract_activations():
    #Extract representations from BERT
    activation_name = 'bert_activations_temp.json'
    transformers_extractor.extract_representations('bert-base-uncased',
        'temp.in',
        activation_name,
        'cuda',
        aggregation="average" #last, first
    )
    return activation_name
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')
    args = parser.parse_args()
    if args.extract == 'True':
        this_activation_name = extract_activations()
    
    activations = load_extracted_activations(this_activation_name)
    print("The length of activations",len(activations))
    print("The activation of the first sentence, length:",activations[0].shape)
    print(activations[0])
    print("The activation of the first sentence, length:",activations[1].shape)
    print(activations[1])

if __name__ == "__main__":
    main()


