import argparse
from utils import Normalization, extract_activations
from utils import get_mappings,all_activations_probe,get_imp_neurons,get_top_words,independent_layerwise_probeing,incremental_layerwise_probeing
from utils import select_independent_neurons,select_minimum_layers
from utils import control_task_probes, probeless,filter_by_frequency,preprocess,alignTokenAct,getOverlap, selectBasedOnTrain
from utils import NumpyEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import collections
import torch
import os
import json


keyword_list = ['False','await','else','import','pass','None','break','except','in','raise','True',
                'class','finally','is','return','and','continue','for','lambda','try','as','def','from',
                'nonlocal','while','assert','del','global','not','with','async''elif','if','or','yield']

keyword_list_train = keyword_list[:17]
keyword_list_valid = keyword_list[17:25]
keyword_list_test = keyword_list[25:]

MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT','pretrained_CodeBERTa','pretrained_UniXCoder',
               'pretrained_RoBERTa','pretrained_JavaBERT','pretrained_GPT2','pretrained_codeGPTJava','pretrained_codeGPTPy',
               'pretrained_codeGPTJavaAdapted','pretrained_codeGPTPyAdapted'
               ]

ACTIVATION_NAMES = {'pretrained_BERT':['bert_activations_train.json','bert_activations_valid.json','bert_activations_test.json'],
                    'pretrained_CodeBERT':['codebert_activations_train.json','codebert_activations_valid.json','codebert_activations_test.json'],
                    'pretrained_GraphCodeBERT':['graphcodebert_activations_train.json','graphcodebert_activations_valid.json','graphcodebert_activations_test.json'],
                    'pretrained_CodeBERTa':['codeberta_activations_train.json','codeberta_activations_valid.json','codeberta_activations_test.json'],
                    "pretrained_UniXCoder":['UniXCoder_activations_train.json','UniXCoder_activations_valid.json','UniXCoder_activations_test.json'],
                    "pretrained_RoBERTa":['RoBERTa_activations_train.json','RoBERTa_activations_valid.json','RoBERTa_activations_test.json'],
                    "pretrained_JavaBERT":['JavaBERT_activations_train.json','JavaBERT_activations_valid.json','JavaBERT_activations_test.json'],
                    "pretrained_GPT2":['GPT2_activations_train.json','GPT2_activations_valid.json','GPT2_activations_test.json'],
                    "pretrained_codeGPTJava":['codeGPTJava_activations_train.json','codeGPTJava_activations_valid.json','codeGPTJava_activations_test.json'],
                    "pretrained_codeGPTPy":['codeGPTPy_activations_train.json','codeGPTPy_activations_valid.json','codeGPTPy_activations_test.json'],
                    "pretrained_codeGPTJavaAdapted":['codeGPTJavaAdapted_activations_train.json','codeGPTJavaAdapted_activations_valid.json','codeGPTJavaAdapted_activations_test.json'],
                    "pretrained_codeGPTPyAdapted":['codeGPTPyAdapted_activations_train.json','codeGPTPyAdapted_activations_valid.json','codeGPTPyAdapted_activations_test.json'],
                    }
AVTIVATIONS_FOLDER = "./activations/"
MODEL_DESC = {"pretrained_BERT":'bert-base-uncased',
              "pretrained_CodeBERT":'microsoft/codebert-base',
              "pretrained_GraphCodeBERT":'microsoft/graphcodebert-base',
              "pretrained_CodeBERTa":'huggingface/CodeBERTa-small-v1',
              "pretrained_UniXCoder":"microsoft/unixcoder-base",
              "pretrained_RoBERTa":"roberta-base",
              "pretrained_JavaBERT":"CAUKiel/JavaBERT",
              "pretrained_GPT2":"gpt2",
              "pretrained_codeGPTJava": "microsoft/CodeGPT-small-java",
              'pretrained_codeGPTPy':"microsoft/CodeGPT-small-py",
              "pretrained_codeGPTJavaAdapted": "microsoft/CodeGPT-small-java-adaptedGPT2",
              "pretrained_codeGPTPyAdapted":"microsoft/CodeGPT-small-py-adaptedGPT2",
              }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')
    parser.add_argument("--this_model", default='pretrained_BERT')

    args = parser.parse_args()
    if args.extract == 'True':
        for this_model in MODEL_NAMES:
            if this_model in ['pretrained_GPT2','pretrained_codeGPTJava','pretrained_codeGPTPy',
                                'pretrained_codeGPTJavaAdapted','pretrained_codeGPTPyAdapted']:
                print(f"Generating the activation file for {this_model}")
                activation_file_name=ACTIVATION_NAMES[this_model][0]
                extract_activations('./src_files/codetest2_train_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
                activation_file_name=ACTIVATION_NAMES[this_model][1]
                extract_activations('./src_files/codetest2_valid_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
                activation_file_name=ACTIVATION_NAMES[this_model][2]
                extract_activations('./src_files/codetest2_test_unique.in',MODEL_DESC[this_model],os.path.join(AVTIVATIONS_FOLDER,activation_file_name))
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )

    torch.manual_seed(0)
    this_model = args.this_model 
    print(f"Anayzing {this_model}")
    tokens_train,activations_train,flat_tokens_train,X_train, y_train, label2idx_train, idx2label_train,_,num_layers=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][0]),
                                                                './src_files/codetest2_train_unique.in','./src_files/codetest2_train_unique.label',
                                                                False,this_model)
    tokens_valid,activations_valid,flat_tokens_valid,X_valid, y_valid, label2idx_valid, idx2label_valid,_,_=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][1]),
                                                    './src_files/codetest2_valid_unique.in','./src_files/codetest2_valid_unique.label',
                                                    False,this_model)
    tokens_test,activations_test,flat_tokens_test,X_test, y_test, label2idx_test, _, sample_idx_test,_=preprocess(os.path.join(AVTIVATIONS_FOLDER,ACTIVATION_NAMES[this_model][2]),
                                    './src_files/codetest2_test_unique.in','./src_files/codetest2_test_unique.label',
                                    False,this_model)
    # remove tokens that are shared by training and testing
    # At the same time, make sure to keep at least 10 key words in the training set
    idx_selected_train = []
    selected_STRING = {}
    selected_NUMBER = {}
    count_kw = 0
    count_number = 0
    count_name = 0
    count_str = 0
    for this_token,this_y in zip(flat_tokens_train,y_train):
        # if this_token in flat_tokens_test:
        if this_y == label2idx_train['NUMBER'] and count_number<=5000 and selected_NUMBER.get(this_token,-1)<=10:
            if this_token in selected_NUMBER:
                selected_NUMBER[this_token] += 1
            else:
                selected_NUMBER[this_token] = 1
            idx_selected_train.append(True)
            count_number+=1
        elif this_token in keyword_list_train and count_kw<=5000:
            idx_selected_train.append(True)
            count_kw+=1
        elif this_y == label2idx_train['STRING'] and count_str<=5000 and selected_STRING.get(this_token,-1)<=2:
            if this_token in selected_STRING:
                selected_STRING[this_token] += 1
            else:
                selected_STRING[this_token] = 1
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
    f = open(f'{this_model}training.txt','w')
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
                                                540)
    print(f"Write tokens in the validation set to files:")
    f = open(f'{this_model}validation.txt','w')
    for this_token in flat_tokens_valid:
        f.write(this_token+"\n")
    f.close()

    X_test, y_test, flat_tokens_test, idx_selected_test, sample_idx_test =selectBasedOnTrain(flat_tokens_test,
                                                                            X_test,
                                                                            y_test,
                                                                            flat_tokens_train,
                                                                            label2idx_train,
                                                                            keyword_list_test,
                                                                            670,
                                                                            sample_idx_test)
    print(f"Write tokens in the testing set to files:")
    f = open(f'{this_model}testing.txt','w')
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

    exit(0)

    neurons_per_layer = X_train.shape[1]//num_layers
    assert neurons_per_layer==X_train.shape[1]/num_layers, f"Model:{this_model},Something is wrong with either number of layers={num_layers} or total neurons={X_train.shape[1]}"
    
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
    all_results['total_neurons'] = X_train.shape[1]

    # All-layer probing
    print("All-layer probing")
    model_name = f"{this_model}_all_layers"
    probe, scores, this_results = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test, y_test,
                                            idx2label_train,tokens_test['source'],model_name,sample_idx_test)
    all_results["baseline"] = this_results
    print("~"*50)

    # Independent-layerwise probing
    print('Independent-layerwise probing')
    results = independent_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                            idx2label_train,tokens_test['source'],this_model,sample_idx_test,num_layers)
    all_results["independent_layerwise"] = results
    print("~"*50)
    # Incremental-layerwise probing
    print('Incremental-layerwise probing')
    results = incremental_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,
                                            idx2label_train,tokens_test['source'],this_model,sample_idx_test,num_layers)
    all_results["incremental_layerwise"] = results
    print("~"*50)
    # select minimum layers
    print('select minimum layers (LS+CC+LCA)')
    target_layer = [0.03,0.02,0.01]
    target_neuron = [0.01]
    clustering_thresholds = [-1,0.3]
    neuron_percentage = [0.001,0.002,0.003,0.004,0.005,0.01,
        0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,
        0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,
        0.70,0.80,0.90,]
    all_results['select_minimum_layer'] = {}
    all_results["select_minimum_neuron"] = {}
    for this_target_layer in target_layer:
        layer_idx = select_minimum_layers(all_results['incremental_layerwise'],this_target_layer,all_results["baseline"]['scores']["__OVERALL__"])
        all_results["select_minimum_layer"][this_target_layer] = layer_idx
        all_results["select_minimum_neuron"][layer_idx] = {}
        # probing using independent neurons based on minimum layers
        for this_target_neuron in target_neuron:
            this_result = select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                    idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,
                    clustering_thresholds,num_layers,neurons_per_layer,this_target_neuron,neuron_percentage,True)
            all_results["select_minimum_neuron"][layer_idx][this_target_neuron] = this_result
    print("~"*50)
    
    # probe independent neurons based on all layers (run_cc_all.py)
    print('probing independent neurons based on all layers (run_cc_all.py)')
    clustering_thresholds = [-1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    layer_idx = num_layers - 1
    this_result = select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                        idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,
                        clustering_thresholds,num_layers,neurons_per_layer,None,None,False)
    all_results["select_from_all_neurons"] = this_result
    print("~"*50)

    # probing independent neurons based on all layers with finer percentage (run_max_features.py)
    print('probing independent neurons based on all layers with finer percentage (run_max_features.py)')
    clustering_thresholds = [-1]
    layer_idx = num_layers - 1
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
                        idx2label_train,label2idx_train,tokens_test['source'],this_model,sample_idx_test,layer_idx,
                        clustering_thresholds,num_layers,neurons_per_layer,this_target_neuron[0],neuron_percentage,True)
    all_results['select_minimum_neurons_finer_percentage'] = this_result
    print("~"*50)

    # Probeless
    probeless_layer_idx = layer_idx
    probeless_neuron_percentage = neuron_percentage
    probeless_target_neuron = this_target_neuron
    this_result = probeless(X_train,y_train,X_valid,y_valid,X_test,y_test,
                            idx2label_train,tokens_test['source'],this_model,
                            sample_idx_test,probeless_layer_idx,num_layers,neurons_per_layer,
                            probeless_target_neuron[0],probeless_neuron_percentage)
    all_results['probeless'] = this_result
    print("~"*50)

    # Important neuron probeing
    top_neurons = get_imp_neurons(probe,label2idx_train,this_model)
    get_top_words(top_neurons,tokens_train,activations_train,this_model)
    del X_train, X_test, X_valid,y_train, y_test,y_valid
    #Control task probes
    selectivity = control_task_probes(flat_tokens_train,X_train_copy,y_train_copy,
                                    flat_tokens_valid, X_valid_copy, y_valid_copy,
                                    flat_tokens_test,X_test_copy,y_test_copy,idx2label_train,scores,this_model,'SAME')
    print("~~~~~~~~~~~~~~~~~~~~~~~Summary~~~~~~~~~~~~~~~~~~~~~~~")
    json_dump = json.dumps(all_results, cls=NumpyEncoder)
    with open(f"{this_model}DetailedOutput.json","w") as f:
        json.dump(json_dump, f)
    f.close()
    
    print(f"Experimental results for {this_model}:")
    print(f"Baseline score (probing using all neurons, {neurons_per_layer} each, of all layers {num_layers}) :{all_results['baseline']['scores']}")
    print()
    print(f"The accuracy when only using the intercept:{all_results['baseline']['intercept']}")
    print()
    print(f"Independent layerwise probing:")
    for i in range(num_layers):
        print(f"Layer {i}:{all_results['independent_layerwise'][f'layer_{i}']['scores']}")
    print()
    print(f"'Incremental-layerwise probing:")
    for i in range(1,num_layers+1):
        layers = list(range(i))
        print(f"Layer {layers}:{all_results['incremental_layerwise'][f'{layers}']['scores']}")

    print()
    best_accuracy = 0
    best_num_neuron = 0
    best_clustering_threshold = -1
    best_layer_idx = -1
    best_target_layer = -1
    best_target_neuron = -1
    best_percent_reduc = 0

    best_lw_layer_idx = -1
    best_lw_target_layer = -1
    best_lw_num_neuron = -1
    best_lw_accuracy = 0
    best_lw_percent_reduc = 0
    print(f"select minimum layers:(LS+CC+LCA)")
    for this_target_layer,layer_idx in all_results['select_minimum_layer'].items():
        print(f"Layerwise (LS):To lose {this_target_layer}*100% accuracy based on all layers, keep the layers from 0 to {layer_idx}")
        neurons2keep = (layer_idx + 1)*neurons_per_layer
        print(f"The number of neurons to keep is {neurons2keep}")
        layers = list(range(layer_idx+1))
        target_layer_accuracy = all_results['incremental_layerwise'][f'{layers}']['scores']
        print(f"The accuracy is:{target_layer_accuracy}")
        lw_percent_reduc = 1 - neurons2keep/all_results['total_neurons']
        print(f"Percentage reduction (neurons):{lw_percent_reduc}")
        print()
        if target_layer_accuracy['__OVERALL__'] > best_lw_accuracy:
            best_lw_layer_idx = layer_idx
            best_lw_target_layer = this_target_layer
            best_lw_accuracy = target_layer_accuracy['__OVERALL__']
            best_lw_percent_reduc =  lw_percent_reduc
            best_lw_num_neuron = neurons2keep

        for this_target_neuron, this_result in all_results['select_minimum_neuron'][layer_idx].items():
            print(f"Clustering based on the layers above: 0 to {layer_idx}:")
            for result_key in this_result:
                if result_key=='no-clustering':
                    print(f"When no clustering:")
                    print(f"the probing result is {this_result[result_key]['base_results']['scores']}")
                    clustering_threshold = -1
                else:
                    clustering_threshold = this_result[result_key]['clustering_threshold']
                    print(f"Clustering threshold:{clustering_threshold}")
                    print(f"The number of independent neurons:{len(this_result[result_key]['independent_neurons'])}")
                    print(f"The number of clusters:{len(this_result[result_key]['clusters'])}")
                    print(f"The probing result (CC score) is :{this_result[result_key]['base_results']['scores']}")

                print(f"To lose {this_target_neuron}*100% of accuracy based on the model above:{result_key}")
                minimal_neuron_set_size = this_result[result_key]['minimal_neuron_set_size']
                print(f"The minimum number of neurons needed is {minimal_neuron_set_size}")
                accuracy = this_result[result_key][f"selected-{minimal_neuron_set_size}-neurons"]['scores']
                print(f"The accuracy of the minimum neuron set is {accuracy}")
                print()
                if accuracy['__OVERALL__'] > best_accuracy:
                    best_accuracy = accuracy['__OVERALL__']
                    best_num_neuron = minimal_neuron_set_size
                    best_percent_reduc = 1 - minimal_neuron_set_size/all_results['total_neurons']
                    best_clustering_threshold = clustering_threshold
                    best_layer_idx = layer_idx
                    best_target_layer = this_target_layer
                    best_target_neuron = this_target_neuron

    print(f"The result of Layerwise (LS):")
    print(f"Keep the layer from 0 to {best_lw_layer_idx}")
    print(f"The best layer delta:{best_lw_target_layer}")
    print(f"The best number of neurons:{best_lw_num_neuron}")
    print(f"The best accuracy:{best_lw_accuracy}")
    print(f"The best percentage reduction: {best_lw_percent_reduc}")
    print()

    print(f"The result of LS+CC+LCA")
    print(f"Keep the layer from 0 to {best_layer_idx}")
    print(f"The best performance delta: {best_target_layer},{best_target_neuron}")
    print(f"The best clustering threshold:{best_clustering_threshold}")
    print(f"The best number of neurons:{best_num_neuron}")
    print(f"The best accuracy: {best_accuracy}")
    print(f"The best neuron percentage reduction: {best_percent_reduc}")
    
    
    print()
    print(f"probe independent neurons based on all layers with clustering (run_cc_all.py)")
    best_accuracy = 0
    best_num_neuron = 0
    best_clustering_threshold = 0
    for result_key in all_results["select_from_all_neurons"]:
        if result_key=='no-clustering':
            print(f"When no clustering:")
            num_neuron = all_results['total_neurons']
            clustering_threshold = -1
        else:
            clustering_threshold = all_results['select_from_all_neurons'][result_key]['clustering_threshold']
            num_neuron = len(all_results['select_from_all_neurons'][result_key]['independent_neurons'])
            print(f"Clustering threshold:{clustering_threshold}")
            print(f"The number of independent neurons:{num_neuron}")
            print(f"The number of clusters:{len(all_results['select_from_all_neurons'][result_key]['clusters'])}")
        
        print(f"The probing result (CC score) is :{all_results['select_from_all_neurons'][result_key]['base_results']['scores']}")
        accuracy = all_results['select_from_all_neurons'][result_key]['base_results']['scores']['__OVERALL__']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_clustering_threshold = clustering_threshold
            best_num_neuron = num_neuron
            percent_reduc = 1 - best_num_neuron/all_results['total_neurons']
    print(f"The result of CC:")
    print(f"The best clustering threshold is :{best_clustering_threshold}")
    print(f"The best number of neurons:{best_num_neuron}")
    print(f"The best accuracy is: {best_accuracy}")
    print(f"Percentage reduction (neurons):{percent_reduc}")
    


    print()
    print(f"probe independent neurons based on all layers without clustering (run_max_features.py)")
    print(f"The result of LCA:")
    result_key='no-clustering'
    target_neuron = all_results['select_minimum_neurons_finer_percentage'][result_key]['target_neuron']
    print(f"Based on all layers: from 0 to {num_layers-1}, no clustering, to lose only {target_neuron}*100% of accuracy:")
    minimum_neuron_size = all_results['select_minimum_neurons_finer_percentage'][result_key]['minimal_neuron_set_size']
    print(f"The minimum number of neurons needed is {minimum_neuron_size}")
    percent_reduc = 1 - minimum_neuron_size/all_results['total_neurons']
    print(f"The performance is {all_results['select_minimum_neurons_finer_percentage'][result_key][f'selected-{minimum_neuron_size}-neurons']}")
    print(f"Percentage reduction (neurons):{percent_reduc}")

    print()
    print(f"Probeless:")
    print(f"The result of probeless:")
    target_neuron = all_results['probeless']['target_neuron']
    print(f"Based on all layers, from 0 to {num_layers-1}, no clustering, to lose only {target_neuron}*100% of accuracy:")
    minimum_neuron_size = all_results['probeless']['probeless_minimal_neuron_set_size']
    print(f"The minimum number of neurons needed is :{minimum_neuron_size}")
    percent_reduc = 1 - minimum_neuron_size/all_results['total_neurons']
    print(f"The performance is :{all_results['probeless'][f'probeless_selected-{minimum_neuron_size}-neurons']}")
    print(f"Percentage reduction (neurons):{percent_reduc}")
    print("----------------------------------------------------------------")
if __name__ == "__main__":
    main()
