import argparse
from utils import Normalization, extract_activations
from utils import get_mappings,all_activations_probe,get_imp_neurons,get_top_words,independent_layerwise_probeing,incremental_layerwise_probeing
from utils import select_independent_neurons
from utils import control_task_probes, probeless,filter_by_frequency,preprocess,alignTokenAct,getOverlap, selectBasedOnTrain
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import torch
import os


keyword_list = ['False','await','else','import','pass','None','break','except','in','raise','True',
                'class','finally','is','return','and','continue','for','lambda','try','as','def','from',
                'nonlocal','while','assert','del','global','not','with','async''elif','if','or','yield']
number = ['0','1','2','3','4','5','6','7','8','9']

keyword_list_train = keyword_list[:17]
keyword_list_valid = keyword_list[17:25]
keyword_list_test = keyword_list[25:]
num_train = number[0:3]
num_valid = number[3:]
num_test = number[3:]

MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',
               'finetuned_defdet_CodeBERT','finetuned_defdet_GraphCodeBERT',
               'finetuned_clonedet_CodeBERT','finetuned_clonedet_GraphCodeBERT']
ACTIVATION_NAMES = {'pretrained_BERT':['bert_activations_train.json','bert_activations_valid.json','bert_activations_test.json'],
                    'pretrained_CodeBERT':['codebert_activations_train.json','codebert_activations_test.json'],
                    'pretrained_GraphCodeBERT':['graphcodebert_activations_train.json','graphcodebert_activations_test.json'],
                    'finetuned_defdet_CodeBERT':['codebert_defdet_activations_train.json','codebert_defdet_activations_test.json'],
                    'finetuned_defdet_GraphCodeBERT':['graphcodebert_defdet_activations_train.json','graphcodebert_defdet_activations_test.json'],
                    'finetuned_clonedet_CodeBERT':['codebert_clonedet_activations1_train.json','codebert_clonedet_activations1_test.json'],
                    'finetuned_clonedet_GraphCodeBERT':['graphcodebert_clonedet_activations1_train.json','graphcodebert_clonedet_activations1_test.json']}
AVTIVATIONS_FOLDER = "./activations/"
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
                extract_activations('./src_files/codetest2_train_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
                activation_file_name=ACTIVATION_NAMES[this_model][1]
                extract_activations('./src_files/codetest2_valid_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
                activation_file_name=ACTIVATION_NAMES[this_model][2]
                extract_activations('./src_files/codetest2_test_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )

    for this_model in MODEL_NAMES:
        torch.manual_seed(0)
        if this_model in ['pretrained_BERT','pretrained_CodeBERT','pretrained_GraphCodeBERT']:
            print(f"Anayzing {this_model}")
            tokens_train,activations_train,flat_tokens_train,X_train, y_train, label2idx_train, idx2label_train,_=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][0]),
                                                                        './src_files/codetest2_train_unique.in','./src_files/codetest2_train_unique.label',
                                                                        False,this_model)
            tokens_valid,activations_valid,flat_tokens_valid,X_valid, y_valid, label2idx_valid, idx2label_valid,_=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][1]),
                                                            './src_files/codetest2_valid_unique.in','./src_files/codetest2_valid_unique.label',
                                                            False,this_model)
            tokens_test,activations_test,flat_tokens_test,X_test, y_test, label2idx_test, _, sample_idx_test=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][2]),
                                            './src_files/codetest2_test_unique.in','./src_files/codetest2_test_unique.label',
                                            False,this_model)
            # remove tokens that are shared by training and testing
            # At the same time, make sure to keep at least 10 key words in the training set
            idx_selected_train = []
            count_kw = 0
            count_number = 0
            count_name = 0
            count_str = 0
            for this_token,this_y in zip(flat_tokens_train,y_train):
                # if this_token in flat_tokens_test:
                if this_y == label2idx_train['NUMBER'] and count_number<=5000:
                    if set(list(this_token)).issubset(num_train):
                        idx_selected_train.append(True)
                        count_number += 1
                    else:
                        idx_selected_train.append(False)
                elif this_token in keyword_list_train and count_kw<=5000:
                    idx_selected_train.append(True)
                    count_kw+=1
                elif this_y == label2idx_train['STRING'] and count_str<=5000:
                    idx_selected_train.append(True)
                    count_str += 1
                elif this_y== label2idx_train['NAME'] and count_name<=5000:
                    idx_selected_train.append(True)
                    count_name += 1
                else:
                    idx_selected_train.append(False)
            assert len(idx_selected_train) == len(flat_tokens_train)

            flat_tokens_train = flat_tokens_train[idx_selected_train]
            X_train = X_train[idx_selected_train]
            y_train = y_train[idx_selected_train]
            tokens_train,activations_train=alignTokenAct(tokens_train,activations_train,idx_selected_train)
            print(f"Write tokens in the training set to files:")
            f = open('training.txt','w')
            for this_token in flat_tokens_train:
                f.write(this_token+"\n")
            f.close()

            assert (flat_tokens_train == np.array([l for sublist in tokens_train['source'] for l in sublist])).all()
            l1 = len([l for sublist in activations_train for l in sublist])
            l2 = len(flat_tokens_train)
            assert l1 == l2,f"{l1}!={l2}"
            assert len(np.array([l for sublist in tokens_train['target'] for l in sublist])) == l2


            X_valid, y_valid, flat_tokens_valid, _, _ =selectBasedOnTrain(flat_tokens_valid,
                                                        X_valid,
                                                        y_valid,
                                                        flat_tokens_train,
                                                        label2idx_train,
                                                        keyword_list_valid,
                                                        num_valid,
                                                        540)
            print(f"Write tokens in the validation set to files:")
            f = open('validation.txt','w')
            for this_token in flat_tokens_valid:
                f.write(this_token+"\n")
            f.close()

            X_test, y_test, flat_tokens_test, idx_selected_test, sample_idx_test =selectBasedOnTrain(flat_tokens_test,
                                                                                    X_test,
                                                                                    y_test,
                                                                                    flat_tokens_train,
                                                                                    label2idx_train,
                                                                                    keyword_list_test,
                                                                                    num_test,
                                                                                    660,
                                                                                    sample_idx_test)
            print(f"Write tokens in the testing set to files:")
            f = open('testing.txt','w')
            for this_token in flat_tokens_test:
                f.write(this_token+"\n")
            f.close()

            tokens_test,_=alignTokenAct(tokens_test,activations_test,idx_selected_test)

            print()
            print("The distribution of classes in training after removing repeated tokens between training and tesing:")
            print(collections.Counter(y_train))
            print(label2idx_train)
            print("The distribution of classes in valid:")
            print(collections.Counter(y_valid))
            print(label2idx_valid)
            print("The distribution of classes in testing:")
            print(collections.Counter(y_test))
            print(label2idx_test)
            
            X_train_copy = X_train.copy()
            y_train_copy = y_train.copy()
            X_valid_copy = X_valid.copy()
            y_valid_copy = y_valid.copy()
            X_test_copy = X_test.copy()
            y_test_copy = y_test.copy()

            #normalize the inputs before doing probing
            norm = Normalization(X_train)
            X_train = norm.norm(X_train)
            X_valid = norm.norm(X_valid)
            X_test = norm.norm(X_test)
            del norm

            all_results={}
            # All-layer probing
            probe, scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test, y_test,
                                                    idx2label_train,tokens_test['source'],this_model,sample_idx_test)
            all_results["baseline"] = scores

            # Independent-layerwise probing
            results = independent_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                                    idx2label_train,tokens_test['source'],this_model,sample_idx_test)
            all_results["independent_layerwise"] = results
            # Incremental-layerwise probing
            results = incremental_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                                    idx2label_train,tokens_test['source'],this_model,sample_idx_test)
            all_results["incremental_layerwise"] = results
            # select minimum layers
            target_layer = [0.03,0.02,0.01]
            target_neuron = [0.01]
            clustering_thresholds = [-1,0.3]
            neuron_percentage = [0.001,0.002,0.003,0.004,0.005,0.01,
                0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,
                0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,
                0.70,0.80,0.90,]
            all_results['select_minimum_layer'] = {}
            for this_target_layer in target_layer:
                layer_idx = select_minimum_layers(all_results['incremental_layerwise'],this_target_layer,all_results["baseline"])
                all_results["select_minimum_layer"][this_target_layer] = layer_idx
                all_results["select_minimum_neuron"][layer_idx] = {}
                # probing using independent neurons based on minimum layers
                for this_target_neuron in target_neuron:
                    this_result = select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                            idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,clustering_thresholds,this_target_neuron,neuron_percentage,True)
                    all_results["select_minimum_neuron"][layer_idx][this_target_neuron] = this_result
            
            # probing independent neurons based on all layers (run_cc_all.py)
            clustering_thresholds = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            layer_idx = 12
            this_result = select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,clustering_thresholds,None,None,False)
            all_results["select_from_all_neurons"] = this_result

            # probing independent neurons based on all layers with coarse percentage (run_max_features.py)
            clustering_thresholds = [-1]
            layer_idx = 12
            this_target_neuron = [0.01]
            neuron_percentage = [0.001,0.002,0.003,0.004,0.005,0.006,
                                0.007,0.008,0.009,0.01,0.011,0.012,
                                0.013,0.014,0.015,0.016,0.017,0.018,
                                0.019,0.02,0.021,0.022,0.023,0.024,
                                0.025,0.026,0.027,0.028,0.029,0.03,
                                0.031,0.032,0.033,0.034,0.035,0.036,
                                0.037,0.038,0.039,0.04,0.041,0.042,
                                0.043,0.044,0.045,0.046,0.047,0.048,
                                0.049,0.05,0.051,0.052,0.053,0.054,
                                0.055,0.056,0.057,0.058,0.059,0.06,
                                0.061,0.062,0.063,0.064,0.065,0.066,
                                0.067,0.068,0.069,0.07,0.071,0.072,
                                0.073,0.074,0.075,0.076,0.077,0.078,
                                0.079,0.08,0.081,0.082,0.083,0.084,
                                0.085,0.086,0.087,0.088,0.089,0.09,
                                0.091,0.092,0.093,0.094,0.095,0.096,
                                0.097,0.098,0.099,0.10,0.15,0.20,0.25,
                                0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,
                                0.90,]
            this_result = select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,clustering_thresholds,this_target_neuron,neuron_percentage,True)
            all_results['select_minimum_neurons_finer_percentage'] = this_result

            # Important neuron probeing
            top_neurons = get_imp_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                            probe,label2idx_train,idx2label_train,tokens_test['source'],this_model,sample_idx_test)
            get_top_words(top_neurons,tokens_train,activations_train,this_model)
            del X_train, X_test, X_valid,y_train, y_test,y_valid
            #Control task probes
            selectivity = control_task_probes(flat_tokens_train,X_train_copy,y_train_copy,
                                            flat_tokens_valid, X_valid_copy, y_valid_copy,
                                            flat_tokens_test,X_test_copy,y_test_copy,idx2label_train,scores,this_model,'SAME')
            print("----------------------------------------------------------------")
            break
if __name__ == "__main__":
    main()
