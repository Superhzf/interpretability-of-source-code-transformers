import argparse
from utils import Normalization, extract_activations, load_extracted_activations, load_tokens
from utils import get_mappings,all_activations_probe,get_imp_neurons,get_top_words,layerwise_probes_inference
from utils import control_task_probes, probeless,filter_by_frequency,preprocess,alignTokenAct,getOverlap
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import torch

weighted = False
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

    for this_model in MODEL_NAMES:
        torch.manual_seed(0)
        if this_model in ['pretrained_BERT','pretrained_CodeBERT','pretrained_GraphCodeBERT']:
            print(f"Anayzing {this_model}")
            tokens_train,activations_train,flat_tokens_train,X_train, y_train, label2idx_train, idx2label_train=preprocess(ACTIVATION_NAMES[this_model][0],
                                                                        './src_files/codetest2_train_unique.in','./src_files/codetest2_train_unique.label',
                                                                        False,this_model)
            tokens_test,activations_test,flat_tokens_test,X_test, y_test, label2idx_test, _=preprocess(ACTIVATION_NAMES[this_model][1],
                                            './src_files/codetest2_test_unique.in','./src_files/codetest2_test_unique.label',
                                            False,this_model)
            # remove tokens that are shared by training and testing
            # At the same time, make sure to keep at least 10 key words in the training set
            idx_selected = []
            count_kw = 0
            count_number = 0
            count_name = 0
            for this_token,this_y in zip(flat_tokens_train,y_train):
                if this_token in flat_tokens_test:
                    if this_y== label2idx_train['NUMBER'] and count_number<=1200:
                        idx_selected.append(True)
                        count_number += 1
                    elif this_y == label2idx_train['KEYWORD'] and count_kw<=1100:
                        idx_selected.append(True)
                        count_kw+=1
                    else:
                        idx_selected.append(False)
                else:
                    if this_y== label2idx_train['NAME']:
                        if count_name <= 1200:
                            count_name += 1
                            idx_selected.append(True)
                        else:
                            idx_selected.append(False)
                    elif this_y== label2idx_train['NUMBER']:
                        if count_number <= 1200:
                            count_number += 1
                            idx_selected.append(True)
                        else:
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

            # This keeps ~10 KW in the test set
            idx_selected = []
            count_number = 0
            count_name = 0
            for this_token_test,this_y_test in zip(flat_tokens_test,y_test):
                if this_token_test in flat_tokens_train:
                    idx_selected.append(False)
                else:
                    # If this_token_test is a key word, then it will be selected for sure.
                    is_selected = True
                    if this_y_test == label2idx_train['STRING']:
                        # Compare this_token_train with this_token_test and remove they are similar (the length of overlap is more than 3)
                        # because it is possible that they are different but very similar. If that is the case,
                        # it is highly likely that the the label would be the same.
                        for this_token_train in flat_tokens_train:
                            if getOverlap(this_token_test,this_token_train) >= 4:
                                is_selected = False
                                break
                    elif this_y_test == label2idx_train['NUMBER']:
                        for this_token_train in flat_tokens_train:
                            if count_number>210 or getOverlap(this_token_test,this_token_train) >= 3:
                                is_selected = False
                                break
                        if is_selected:
                            count_number += 1
                    elif this_y_test == label2idx_train['NAME']:
                        for this_token_train in flat_tokens_train:
                            if count_name> 210 or getOverlap(this_token_test,this_token_train) >= 2:
                                is_selected = False
                                break
                        if is_selected:
                            count_name += 1
                    idx_selected.append(is_selected)
            assert len(idx_selected) == len(flat_tokens_test)
            flat_tokens_test = flat_tokens_test[idx_selected]
            X_test = X_test[idx_selected]
            y_test = y_test[idx_selected]
            tokens_test,_=alignTokenAct(tokens_test,activations_test,idx_selected)

            print()
            print("The distribution of classes in training after removing repeated tokens between training and tesing:")
            print(collections.Counter(y_train))
            print(label2idx_train)
            print("The distribution of classes in testing:")
            print(collections.Counter(y_test))
            print(label2idx_test)
            
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
            X_test_copy = X_test.copy()
            y_test_copy = y_test.copy()

            X_train, X_valid, y_train, y_valid = \
                train_test_split(X_train, y_train, test_size=0.15, shuffle=False)

            #normalize the inputs before doing probing
            norm = Normalization(X_train)
            X_train = norm.norm(X_train)
            X_valid = norm.norm(X_valid)
            X_test = norm.norm(X_test)
            del norm

            #Probeless clustering experiments
            # probeless(X_train,y_train,this_model)

            #All activations probes
            print()
            print("The shape of the training set:",X_train.shape)
            print("The shape of the validation set:",X_valid.shape)
            print("The shape of the testing set:",X_test.shape)
            probe, scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test, y_test,
                                                    idx2label_train,tokens_test['source'],weighted,this_model)

            #Layerwise Probes
            layerwise_probes_inference(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                                    idx2label_train,tokens_test['source'],weighted,this_model)

            #Important neuron probes
            top_neurons = get_imp_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                            probe,label2idx_train,idx2label_train,tokens_test['source'],weighted,this_model)
            get_top_words(top_neurons,tokens_train,activations_train,this_model)
            del X_train, X_test, X_valid,y_train, y_test,y_valid
            #Control task probes
            selectivity = control_task_probes(flat_tokens_train,X_train_copy,y_train_copy,
                                            flat_tokens_test,X_test_copy,y_test_copy,idx2label_train,scores,this_model,'SAME')
            print("----------------------------------------------------------------")
            break
if __name__ == "__main__":
    main()
