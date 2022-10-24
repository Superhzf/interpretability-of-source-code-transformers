import argparse
from utils import remove_seen_tokens,Normalization, extract_activations, load_extracted_activations, load_tokens
from utils import get_mappings,all_activations_probe,get_imp_neurons,get_top_words,layerwise_probes_inference
from utils import control_task_probes, probeless,filter_by_frequency,preprocess,alignTokenAct
from sklearn.model_selection import train_test_split
import numpy as np
import collections

MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',
               'finetuned_defdet_CodeBERT','finetuned_defdet_GraphCodeBERT',
               'finetuned_clonedet_CodeBERT','finetuned_clonedet_GraphCodeBERT']
# ACTIVATION_NAMES = ['bert_activations.json',
#                     'codebert_activations.json','graphcodebert_activations.json',
#                     'codebert_defdet_activations.json','graphcodebert_defdet_activations.json',
#                     'codebert_clonedet_activations1.json','graphcodebert_clonedet_activations1.json']
ACTIVATION_NAMES = {'pretrained_BERT':['bert_activations_train.json','bert_activations_test.json'],
                    'pretrained_CodeBERT':['codebert_activations_train.json','codebert_activations_test.json'],
                    'pretrained_GraphCodeBERT':['graphcodebert_activations_train.json','graphcodebert_activations_test.json'],
                    'finetuned_defdet_CodeBERT':['codebert_defdet_activations_train.json','codebert_defdet_activations_test.json'],
                    'finetuned_defdet_GraphCodeBERT':['graphcodebert_defdet_activations_train.json','graphcodebert_defdet_activations_test.json'],
                    'finetuned_clonedet_CodeBERT':['codebert_clonedet_activations1_train.json','codebert_clonedet_activations1_test.json'],
                    'finetuned_clonedet_GraphCodeBERT':['graphcodebert_clonedet_activations1_train.json','graphcodebert_clonedet_activations1_test.json']}
MODEL_DESC = {"pretrained_BERT":'bert-base-uncased',
              "pretrained_CodeBERT":'microsoft/codebert-base',
              "pretrained_GraphCodeBERT":'microsoft/graphcodebert-base'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')

    args = parser.parse_args()
    if args.extract == 'True':
        for this_model in MODEL_NAMES:
            if this_model in ['pretrained_BERT','pretrained_CodeBERT','pretrained_GraphCodeBERT']:
                print(f"Generating the activation file for {this_model}")
                activation_file_name=ACTIVATION_NAMES[this_model][0]
                extract_activations('./src_files/codetest2_train_unique.in',MODEL_DESC[this_model],activation_file_name)
                activation_file_name=ACTIVATION_NAMES[this_model][1]
                extract_activations('./src_files/codetest2_test_unique.in',MODEL_DESC[this_model],activation_file_name)
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )
    exit(0)

    for this_model in MODEL_NAMES:
        if this_model in ['pretrained_BERT','pretrained_CodeBERT','pretrained_GraphCodeBERT']:
            print(f"Anayzing {this_model}")
            freq_threshold = 3
            tokens_train,activations_train,flat_tokens_train,X_train, y_train, label2idx_train, idx2label_train=preprocess(ACTIVATION_NAMES[this_model][0],
                                                                        './src_files/codetest2_train_unique.in','./src_files/codetest2_train_unique.label',
                                                                        freq_threshold,this_model)
            _,_,flat_tokens_test,X_test, y_test, _, _=preprocess(ACTIVATION_NAMES[this_model][1],
                                            './src_files/codetest2_test_unique.in','./src_files/codetest2_test_unique.label',
                                            freq_threshold,this_model)
            # remove tokens that are shared by training and testing
            idx_selected = []
            for this_token in flat_tokens_train:
                if this_token in flat_tokens_test:
                    idx_selected.append(False)
                else:
                    idx_selected.append(True)
            assert len(idx_selected) == len(flat_tokens_train)

            flat_tokens_train = flat_tokens_train[idx_selected]
            X_train = X_train[idx_selected]
            y_train = y_train[idx_selected]
            tokens_train,activations_train=alignTokenAct(tokens_train,activations_train,idx_selected)

            assert (flat_tokens_train == np.array([l for sublist in tokens_train['source'] for l in sublist])).all()
            l1 = len([l for sublist in activations_train for l in sublist])
            l2 = len(flat_tokens_train)
            assert l1 == l2,f"{l1}!={l2}"
            assert len(np.array([l for sublist in tokens_train['target'] for l in sublist])) == l2

            print("The distribution of classes in training after removing repeated tokens between training and tesing:")
            print(collections.Counter(y_train))
            print(label2idx_train)

            lookup_table = {1:2,2:3,3:1}
            for idx, this_y in enumerate(y_test):
                if this_y in lookup_table:
                    y_test[idx] = lookup_table[this_y]
            
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
            X_test_copy = X_test.copy()
            y_test_copy = y_test.copy()

            X_train, X_valid, y_train, y_valid = \
                train_test_split(X_train, y_train, test_size=0.1, shuffle=False)

            #normalize the inputs before doing probing
            norm = Normalization(X_train)
            X_train = norm.norm(X_train)
            X_valid = norm.norm(X_valid)
            X_test = norm.norm(X_test)
            del norm

            #Probeless clustering experiments
            # probeless(X_train,y_train,this_model)

            #All activations probes
            print("The shape of the training set:",X_train.shape)
            print("The shape of the validation set:",X_valid.shape)
            print("The shape of the testing set:",X_test.shape)
            probe, scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test, y_test,idx2label_train,this_model)

            #Layerwise Probes
            layerwise_probes_inference(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label_train,this_model)

            #Important neuron probes
            top_neurons = get_imp_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,probe,label2idx_train,idx2label_train,this_model)
            get_top_words(top_neurons,tokens_train,activations_train,this_model)
            del X_train, X_test, X_valid,y_train, y_test,y_valid
            #Control task probes
            selectivity = control_task_probes(flat_tokens_train,X_train_copy,y_train_copy,
                                            flat_tokens_test,X_test_copy,y_test_copy,idx2label_train,scores,this_model,'UNIFORM')
            print("----------------------------------------------------------------")

if __name__ == "__main__":
    main()
