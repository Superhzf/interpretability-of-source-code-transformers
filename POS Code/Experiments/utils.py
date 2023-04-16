import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
# /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/NeuroX/neurox/data/loader.py
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation
import neurox.data.control_task as ct
import neurox.interpretation.clustering as clustering
import neurox.interpretation.probeless as neuronx_probeless
from sklearn.model_selection import train_test_split
import neurox.analysis.corpus as corpus
import numpy as np
import collections
import difflib
import torch
from transformers import AutoTokenizer, AutoModel
import json

l1 = [0,1e-5,1e-4,1e-3,1e-2,0.1]
l2 = [0,1e-5,1e-4,1e-3,1e-2,0.1]

def getOverlap(s1, s2):
    try:
        s1 = s1.lower()
        s2 = s2.lower()
    except:
        pass
    s = difflib.SequenceMatcher(None, s1, s2)
    pos_a, pos_b, size = s.find_longest_match(0, len(s1), 0, len(s2)) 
    return len(s1[pos_a:pos_a+size])


def removeSeenTokens(tokens,activations):
    """
    Remove the duplicated tokens and the corresponded representation.
    This will not affect the grammar because this is executed after the representation is generated.
    """
    seen_before = []
    new_source_tokens = []
    new_target_tokens = []
    new_activations = []

    source_tokens = tokens['source']
    target_tokens = tokens['target']
    for obs_idx,this_obs in enumerate(source_tokens):
        this_source = []
        this_target = []
        this_activation = []
        for token_idx,this_token in enumerate(this_obs):
            if this_token not in seen_before:
                seen_before.append(this_token)
                this_source.append(this_token)
                this_target.append(target_tokens[obs_idx][token_idx])
                this_activation.append(activations[obs_idx][token_idx])
        assert len(this_source) == len(this_target)
        assert len(this_source) == len(this_activation)
        if len(this_source)>0:
            this_source = np.array(this_source)
            this_target = np.array(this_target)
            this_activation = np.array(this_activation)
            new_source_tokens.append(this_source)
            new_target_tokens.append(this_target)
            new_activations.append(this_activation)
    new_tokens = {"source":new_source_tokens,"target":new_target_tokens}
    return new_tokens,new_activations


class Normalization:
    def __init__(self,df):
        self.var_mean = np.mean(df,axis=0)
        self.var_std = np.std(df,axis=0)

    def norm(self,df):
        norm_df = (df-self.var_mean)/self.var_std
        return norm_df


#Extract activations.json files
def extract_activations(file_in_name,model_description,activation_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transformers_extractor.extract_representations(model_description,
        file_in_name,
        activation_name,
        device,
        aggregation="average" #last, first
    )


def load_extracted_activations(activation_file_name):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(activation_file_name)
    return activations, num_layers


def load_tokens(activations,FILES_IN,FILES_LABEL):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens, sample_idx = data_loader.load_data(FILES_IN,
                                   FILES_LABEL,
                                   activations,
                                   512 # max_sent_length
                                  )
    return tokens, sample_idx


def param_tuning(X_train,y_train,X_valid,y_valid,idx2label,l1,l2):
    best_l1 = None
    best_l2 = None
    best_score = -float('inf')
    best_probe = None
    for this_l1 in l1:
        for this_l2 in l2:
            this_probe = linear_probe.train_logistic_regression_probe(X_train, y_train,
                                                                    lambda_l1=this_l1,
                                                                    lambda_l2=this_l2,
                                                                    num_epochs=10,
                                                                    batch_size=128)
            this_score = linear_probe.evaluate_probe(this_probe, X_valid, y_valid, idx_to_class=idx2label)
            if this_score['__OVERALL__'] > best_score:
                best_score = this_score['__OVERALL__']
                best_l1 = this_l1
                best_l2 = this_l2
                best_probe = this_probe
        return best_l1,best_l2,best_probe


def get_mappings(tokens,activations):
    '''Re-organize the representation and labels such that they are ready for model training'''
    X, y, mapping = utils.create_tensors(tokens, activations, 'NAME') #mapping contains tuple of 4 dictionaries
    label2idx, idx2label, src2idx, idx2src = mapping

    return X, y, label2idx, idx2label, src2idx, idx2src


def all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,src_tokens_test,model_name,sample_idx_test=None,need_cm=False):
    #Train the linear probes (logistic regression) - POS(code) tagging
    best_l1,best_l2,best_probe=param_tuning(X_train,y_train,X_valid,y_valid,idx2label,l1,l2)
    #Get scores of probes
    results = {}
    scores,predictions = linear_probe.evaluate_probe(best_probe, X_test, y_test,idx_to_class=idx2label,
                                                    return_predictions=True,source_tokens=src_tokens_test)
    results['model_name'] = model_name
    results['best_l1'] = best_l1
    results['best_l2'] = best_l2
    results['scores'] = scores
    if src_tokens_test is not None and need_cm:
        NAME_NAME, NAME_KW, NAME_STRING,NAME_NUMBER, KW_NAME, KW_KW, KW_other= 0, 0, 0, 0, 0, 0, 0
        NAME_STRING_list,NAME_NUMBER_list = [], []
        NAME_NAME_list = []
        NAME_NUMBER_samples = []
        NAME_NAME_samples = []
        for idx,this_y_test in enumerate(y_test):
            predicted_class = predictions[idx][1]
            source_token = predictions[idx][0]
            sample = sample_idx_test[idx]
            if idx2label[this_y_test] == "NAME":
                if predicted_class == 'NAME':
                    NAME_NAME += 1
                    NAME_NAME_samples.append(sample)
                    NAME_NAME_list.append(source_token)
                elif predicted_class == 'KEYWORD':
                    NAME_KW += 1
                elif predicted_class == 'STRING':
                    NAME_STRING += 1
                    NAME_STRING_list.append(source_token)
                elif predicted_class == 'NUMBER':
                    NAME_NUMBER += 1
                    NAME_NUMBER_list.append(source_token)
                    NAME_NUMBER_samples.append(sample)
            elif idx2label[this_y_test] == "KEYWORD":
                if predicted_class == 'KEYWORD':
                    KW_KW += 1
                elif predicted_class == 'NAME':
                    KW_NAME += 1
                else:
                    KW_other += 1
        
        results['NAME_NAME'] = NAME_NAME
        results['KW_NAME'] = KW_NAME
        results['NAME_NAME_list'] = NAME_NAME_list
        results['NAME_NAME_sample'] = NAME_NAME_samples
        results['NAME_KW'] = NAME_KW
        results['KW_KW'] = KW_KW
        results['NAME_STRING'] = NAME_STRING
        results['KW_other'] = KW_other
        results['NAME_NUMBER'] = NAME_NUMBER
        results['NAME_STRING_list'] = NAME_STRING_list
        results['NAME_NUMBER_list'] = NAME_NUMBER_list
        results['NAME_NUMBER_sample'] = NAME_NUMBER_samples
    X_test_baseline = np.zeros_like(X_test)

    scores_intercept = linear_probe.evaluate_probe(best_probe, X_test_baseline, y_test, idx_to_class=idx2label)
    results['intercept'] = scores_intercept
    return best_probe, scores, results


def get_imp_neurons(probe,label2idx,model_name):
    ''' Returns top 2% neurons for each model'''

    #Top neurons
    # 0.05 means to select neurons that take top 5% mass
    top_neurons, top_neurons_per_class = linear_probe.get_top_neurons(probe, 0.05, label2idx)
    print()
    print(f"{model_name} top neurons")
    print(repr(top_neurons))
    print(f"{model_name} top neurons per class")
    print(top_neurons_per_class)

    return top_neurons


def get_top_words(top_neurons,tokens,activations,model_name):
    #relate neurons to corpus elements like words and sentences
    print(f"{model_name} top words")
    for neuron in top_neurons:
        top_words = corpus.get_top_words(tokens, activations, neuron, num_tokens=5)
        print(f"Top words for {model_name} neuron indx {neuron}",top_words)


def independent_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,src_tokens_test,model_name,sample_idx_test,num_layers):
    ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''
    results = {}
    need_cm = True
    for i in range(num_layers):
        this_model_name = f"{model_name}_layer_{i}"
        layer_train = ablation.filter_activations_by_layers(X_train, [i], num_layers)
        layer_valid = ablation.filter_activations_by_layers(X_valid, [i], num_layers)
        layer_test = ablation.filter_activations_by_layers(X_test, [i], num_layers)
        _,this_score,this_result = all_activations_probe(layer_train,y_train,layer_valid,y_valid,layer_test,y_test,
                                    idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)
        results[f"layer_{i}"] = this_result
    return results


def incremental_layerwise_probeing(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,src_tokens_test,model_name,sample_idx_test,num_layers):
    ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''
    results = {}
    need_cm = True
    for i in range(1,num_layers+1):
        layers = list(range(i))
        this_model_name = f"{model_name}_layer_{layers}"
        layer_train = ablation.filter_activations_by_layers(X_train, layers, num_layers)
        layer_valid = ablation.filter_activations_by_layers(X_valid, layers, num_layers)
        layer_test = ablation.filter_activations_by_layers(X_test, layers, num_layers)
        _,this_score,this_result = all_activations_probe(layer_train,y_train,layer_valid,y_valid,layer_test,y_test,
                                    idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)
        results[f"{layers}"] = this_result
    return results


def select_minimum_layers(incremental_layerwise_result,target,all_layer_result):
    """
    Select the minimum number of layers such that the performance fullfil the target
    """
    for idx, this_accuracy in enumerate(list(incremental_layerwise_result.values())):
        if this_accuracy['scores']["__OVERALL__"] > all_layer_result*(1-target):
            return idx
    assert 1==0, "something is wrong"


def select_independent_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,
                            idx2label,label2idx,src_tokens_test,this_model_name,
                            sample_idx_test,layer_idx,clustering_threshold,num_layers,neurons_per_layer,
                            target_neuron=None,neuron_percentage=None,full_probing=False):
    result = {}
    layers = list(range(layer_idx+1))
    layer_train = ablation.filter_activations_by_layers(X_train, layers, num_layers)
    layer_valid = ablation.filter_activations_by_layers(X_valid, layers, num_layers)
    layer_test = ablation.filter_activations_by_layers(X_test, layers, num_layers)
    for this_threshold in clustering_threshold:
        if this_threshold == -1:
            X_train_filtered = layer_train
            X_valid_filtered = layer_valid
            X_test_filtered = layer_test
            result_key = "no-clustering"
            result[result_key]={}
            need_cm = False
        else:
            need_cm = True
            result_key = f"clustering-{this_threshold}"
            result[result_key] = {}
            independent_neurons, clusters = clustering.extract_independent_neurons(layer_train, use_abs_correlation=True, clustering_threshold=this_threshold)
            X_train_filtered = ablation.filter_activations_keep_neurons(layer_train,independent_neurons)
            X_valid_filtered = ablation.filter_activations_keep_neurons(layer_valid,independent_neurons)
            X_test_filtered = ablation.filter_activations_keep_neurons(layer_test,independent_neurons)
            result[result_key]["clustering_threshold"] = this_threshold
            result[result_key]["clusters"] = [int(x) for x in clusters]
            result[result_key]["independent_neurons"] = [int(x) for x in independent_neurons]

        model,this_score,this_result = all_activations_probe(X_train_filtered,y_train,X_valid_filtered,y_valid,X_test_filtered,y_test,
                                    idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)
        result[result_key]['base_results'] = this_result
        result[result_key]['target_neuron'] = target_neuron

        if full_probing:
            need_cm = False
            target_score = this_score["__OVERALL__"]*(1-target_neuron)
            ordering,_ = linear_probe.get_neuron_ordering(model, label2idx,search_stride=1000)
            result[result_key]["ordering"] = [int(x) for x in ordering]
            minimal_neuron_set_size = X_train_filtered.shape[1]
            result[result_key]["minimal_neuron_set_size"] = minimal_neuron_set_size
            result[result_key]["minimal_neuron_set"] = result[result_key]["ordering"]
            result[result_key][f"selected-{X_train_filtered.shape[1]}-neurons"] = this_result
            for this_percentage in neuron_percentage:
                selected_num_neurons = int(this_percentage * num_layers * neurons_per_layer)
                if selected_num_neurons > X_train_filtered.shape[1]:
                    continue
                selected_neurons = ordering[:selected_num_neurons]
                X_train_selected = ablation.filter_activations_keep_neurons(X_train_filtered, selected_neurons)
                X_valid_selected = ablation.filter_activations_keep_neurons(X_valid_filtered, selected_neurons)
                X_test_selected = ablation.filter_activations_keep_neurons(X_test_filtered, selected_neurons)
                _,this_score, this_result = all_activations_probe(X_train_selected,y_train,X_valid_selected,y_valid,X_test_selected,y_test,
                                        idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)
                result[result_key][f"selected-{selected_num_neurons}-neurons"] = this_result
                if this_score["__OVERALL__"] > target_score:
                    minimal_neuron_set_size = selected_num_neurons
                    result[result_key]["minimal_neuron_set_size"] = minimal_neuron_set_size
                    result[result_key]["minimal_neuron_set"] = [int(x) for x in selected_neurons]
                    break
    return result



def randomReassignment(tokens,labels,distribution):
    lookup_table={}
    #random assign new class
    # for this_class in label_freqs.keys():
    #     lookup_table[this_class] = np.random.choice(list(label_freqs.keys()), p=distribution)

    for idx,this_token in enumerate(tokens):
            if this_token not in lookup_table:
                np.random.seed(idx)
                lookup_table[this_token] = np.random.choice(labels, p=distribution)
    y_ct = []
    for this_token in tokens:
        this_y_ct = lookup_table[this_token]
        y_ct.append(this_y_ct)
    return y_ct


def control_task_probes(tokens_train,X_train,y_train,tokens_valid,X_valid,y_valid,tokens_test,X_test,y_test,idx2label_train,original_scores,model_name,method):
    print(f"Creating control dataset for {model_name} POS tagging task")
    need_cm = False
    label_freqs = collections.Counter(y_train)
    distribution = []
    if method == 'SAME':
        total = sum(label_freqs.values())
        for this_class,freq in label_freqs.items():
            distribution.append(freq/total)
    elif method == "UNIFORM":
        for this_class,freq in label_freqs.items():
            distribution.append(1/len(label_freqs))
    else:
        assert 1==0, "method is not understood"
    while True:
        y_train_ct = randomReassignment(tokens_train,list(label_freqs.keys()),distribution)
        y_valid_ct = randomReassignment(tokens_valid,list(label_freqs.keys()),distribution)
        y_test_ct = randomReassignment(tokens_test,list(label_freqs.keys()),distribution)
        assert len(y_train_ct) == len(y_train)
        assert len(y_valid_ct) == len(y_valid)
        assert len(y_test_ct) == len(y_test)
        y_train_ct = np.array(y_train_ct)
        y_valid_ct = np.array(y_valid_ct)
        y_test_ct = np.array(y_test_ct)

        # X_train_ct, X_valid_ct, y_train_ct, y_valid_ct = \
        #     train_test_split(X_train, y_train_ct, test_size=0.1, shuffle=False)
        # class 0,1,2 must be in y_train_ct
        if 0 in y_train_ct and 1 in y_train_ct and 2 in y_train_ct:
            break
    y_train = y_train_ct
    y_valid = y_valid_ct
    y_test = y_test_ct
    del y_train_ct,y_valid_ct,y_test_ct
    # normalization
    ct_norm = Normalization(X_train)
    X_train = ct_norm.norm(X_train)
    X_valid = ct_norm.norm(X_valid)
    X_test = ct_norm.norm(X_test)
    del ct_norm

    assert X_train.shape[0] == len(y_train)
    assert X_valid.shape[0] == len(y_valid)
    assert X_test.shape[0] == len(y_test)
    model_name = f'{model_name}_control_task'
    _, ct_scores, this_result = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label_train,None,model_name, None,need_cm)
    
    selectivity = original_scores['__OVERALL__'] - ct_scores['__OVERALL__']
    print()
    print(f'{model_name} Selectivity (Diff. between true task and probing task performance): ', selectivity)
    del ct_scores
    return selectivity


def probeless(X_train,y_train,X_valid,y_valid,X_test,y_test,
                idx2label,src_tokens_test,this_model_name,
                sample_idx_test,layer_idx,num_layers,neurons_per_layer,
                target_neuron,neuron_percentage):
    '''General and Task specific probeless '''
    result = {}
    need_cm = False
    layers = list(range(layer_idx+1))
    layer_train = ablation.filter_activations_by_layers(X_train, layers, num_layers)
    layer_valid = ablation.filter_activations_by_layers(X_valid, layers, num_layers)
    layer_test = ablation.filter_activations_by_layers(X_test, layers, num_layers)

    model,this_score,this_result = all_activations_probe(layer_train,y_train,layer_valid,y_valid,layer_test,y_test,
                                    idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)

    overall_ordering, ordering_per_tag = neuronx_probeless.get_neuron_ordering_for_all_tags(X_train,y_train,idx2label)
    result['probeless_overall_ordering'] = overall_ordering
    result['probeless_ordering_per_tag'] = ordering_per_tag
    result['target_neuron'] = target_neuron
    
    target_score = this_score["__OVERALL__"]*(1-target_neuron)
    minimal_neuron_set_size = layer_train.shape[1]
    result[f"probeless_minimal_neuron_set_size"] = minimal_neuron_set_size
    result[f"probeless_minimal_neuron_set"] = result["probeless_overall_ordering"]
    result[f"probeless_selected-{layer_train.shape[1]}-neurons"] = this_result
    for this_percentage in neuron_percentage:
        selected_num_neurons = int(this_percentage * num_layers * neurons_per_layer)
        if selected_num_neurons > layer_train.shape[1]:
            continue
        selected_neurons = overall_ordering[:selected_num_neurons]
        X_train_selected = ablation.filter_activations_keep_neurons(layer_train, selected_neurons)
        X_valid_selected = ablation.filter_activations_keep_neurons(layer_valid, selected_neurons)
        X_test_selected = ablation.filter_activations_keep_neurons(layer_test, selected_neurons)
        _,this_score, this_result = all_activations_probe(X_train_selected,y_train,X_valid_selected,y_valid,X_test_selected,y_test,
                                idx2label,src_tokens_test,this_model_name,sample_idx_test,need_cm)
        result[f"probeless_selected-{selected_num_neurons}-neurons"] = this_result
        if this_score["__OVERALL__"] > target_score:
            minimal_neuron_set_size = selected_num_neurons
            result["probeless_minimal_neuron_set_size"] = minimal_neuron_set_size
            result["probeless_minimal_neuron_set"] = [int(x) for x in selected_neurons]
            break
    return result



def alignTokenAct(tokens,activations,idx_selected):
    """
    This method means to filter tokens and activations by idx_selected while keeping the same format.
    """
    l1 = len([l for sublist in activations for l in sublist])
    l2 = len(idx_selected)
    assert l1 == l2,f"{l1}!={l2}"
    new_tokens_src = []
    new_tokens_trg = []
    new_activations = []
    idx = 0
    for this_tokens_src,this_tokens_trg,this_activations in zip(tokens['source'],tokens['target'],activations):
        this_new_tokens_src = []
        this_new_tokens_trg = []
        this_new_activations = []
        for this_token_src,this_token_trg,this_activation in zip(this_tokens_src,this_tokens_trg,this_activations):
            if idx_selected[idx]:
                this_new_tokens_src.append(this_token_src)
                this_new_tokens_trg.append(this_token_trg)
                this_new_activations.append(this_activation)
            idx+=1
        if len(this_new_tokens_src)>0:
            this_new_tokens_src = np.array(this_new_tokens_src)
            this_new_tokens_trg = np.array(this_new_tokens_trg)
            this_new_activations = np.array(this_new_activations)
            new_tokens_src.append(this_new_tokens_src)
            new_tokens_trg.append(this_new_tokens_trg)
            new_activations.append(this_new_activations)
    assert idx == len(idx_selected)
    new_tokens = {'source':new_tokens_src,'target':new_tokens_trg}
    return new_tokens,new_activations


def filter_by_frequency(tokens,activations,X,y,label2idx,idx2label,threshold,model_name):
    count = collections.Counter(y)
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print()
    print(f"{model_name} distribution:")
    print(distribution)

    flat_src_tokens = np.array([l for sublist in tokens['source'] for l in sublist])
    assert len(flat_src_tokens) == len(y)
    idx_selected = y <= threshold
    y = y[idx_selected]
    X = X[idx_selected]
    flat_src_tokens = flat_src_tokens[idx_selected]
    tokens,activations=alignTokenAct(tokens,activations,idx_selected)
    assert (flat_src_tokens == np.array([l for sublist in tokens['source'] for l in sublist])).all()
    l1 = len([l for sublist in activations for l in sublist])
    l2 = len(flat_src_tokens)
    assert l1 == l2,f"{l1}!={l2}"
    assert len(np.array([l for sublist in tokens['target'] for l in sublist])) == l2

    label2idx = {label:idx for (label,idx) in label2idx.items() if idx <= threshold}
    idx2label = {idx:label for (idx,label) in idx2label.items() if idx <= threshold}

    count = collections.Counter(y)
    distribution_rate = {k: v/len(y) for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print(f"{model_name} distribution after trauncating:")
    print(distribution_rate)
    print(distribution)
    print(label2idx)
    return tokens,activations,flat_src_tokens,X,y,label2idx,idx2label


def filterByClass(tokens,activations,X,y,label2idx,model_name,sample_idx,class_wanted):
    """
    This method means to keep the representation and labels for
    NAME, STRING, NUMBER, and KEYWORD class for the probing task.
    """
    lookup_table={}
    new_label2idx={}
    new_idx2label={}

    flat_targt_tokens = np.array([l for sublist in tokens['target'] for l in sublist])
    flat_src_tokens = np.array([l for sublist in tokens['source'] for l in sublist])
    flat_sample_idx = np.array([[idx,idxInCode] for idx,sublist in zip(sample_idx,tokens['source']) for idxInCode,l in enumerate(sublist)])
    assert len(flat_targt_tokens) == len(y)
    assert len(flat_src_tokens) == len(y)
    assert len(flat_sample_idx) == len(y)

    
    for idx,this_class in enumerate(class_wanted):
        lookup_table[label2idx[this_class]] = idx
        new_label2idx[this_class] = idx
        new_idx2label[idx] = this_class
    idx_selected=[]
    for this_targt in flat_targt_tokens:
        if this_targt in class_wanted:
            idx_selected.append(True)
        else:
            idx_selected.append(False)
    y = y[idx_selected]
    y = [lookup_table[this_y] for this_y in y]
    y = np.array(y)
    X = X[idx_selected]
    flat_sample_idx = flat_sample_idx[idx_selected]

    flat_src_tokens = flat_src_tokens[idx_selected]
    tokens,activations=alignTokenAct(tokens,activations,idx_selected)
    assert (flat_src_tokens == np.array([l for sublist in tokens['source'] for l in sublist])).all()
    l1 = len([l for sublist in activations for l in sublist])
    l2 = len(flat_src_tokens)
    assert l1 == l2,f"{l1}!={l2}"
    assert len(np.array([l for sublist in tokens['target'] for l in sublist])) == l2

    count = collections.Counter(y)
    distribution_rate = {k: v/len(y) for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print(f"{model_name} distribution after trauncating:")
    print(distribution_rate)
    print(distribution)
    print(new_label2idx)
    return tokens,activations,flat_src_tokens,X,y,new_label2idx,new_idx2label, flat_sample_idx


def preprocess(activation_file_name,IN_file,LABEL_file,remove_seen_tokens,model_name,class_wanted):
    activations,num_layers = load_extracted_activations(activation_file_name)
    tokens, sample_idx =  load_tokens(activations,IN_file,LABEL_file)
    if remove_seen_tokens:
        tokens,activations=removeSeenTokens(tokens,activations)
    X, y, label2idx, _, _, _ = get_mappings(tokens,activations)
    tokens,activations,flat_src_tokens,X_train, y_train, label2idx, idx2label, sample_idx = filterByClass(tokens,activations,X,y,label2idx,model_name,sample_idx,class_wanted)
    return tokens,activations,flat_src_tokens,X_train,y_train,label2idx,idx2label, sample_idx, num_layers


def selectBasedOnTrain(flat_tokens_test,X_test, y_test,flat_tokens_train,label2idx_test,idx2label_test,special_class_split,upper_bound,sample_idx_test=None):
    idx_selected = []
    counter = {}
    for label,index in label2idx_test.items():
        counter[index] = 0
    for this_token_test,this_y_test in zip(flat_tokens_test,y_test):
        if this_token_test in flat_tokens_train:
            idx_selected.append(False)
        else:
            is_selected = True
            if idx2label_test[this_y_test] in special_class_split.values():
                this_class = idx2label_test[this_y_test]
                if this_y_test not in special_class_split[this_class] or counter[this_y_test] >= upper_bound:
                    is_selected = False
                else:
                    counter[this_y_test] += 1
            else:
                if counter[this_y_test] >= upper_bound:
                    is_selected = False
                else:
                    counter[this_y_test] += 1
            idx_selected.append(is_selected)
    assert len(idx_selected) == len(flat_tokens_test)
    flat_tokens_test = flat_tokens_test[idx_selected]
    X_test = X_test[idx_selected]
    y_test = y_test[idx_selected]
    if sample_idx_test is not None:
        sample_idx_test = sample_idx_test[idx_selected]
        assert len(sample_idx_test) == len(y_test)
    return X_test, y_test, flat_tokens_test, idx_selected, sample_idx_test

def selectTrain(flat_tokens_train,y_train,unique_token_label_train,unique_token_label_valid,unique_token_label_test,
                label2idx_train,idx2label_train,priority_list=None):
    idx_selected_train = []
    counter = {}
    if priority_list is not None:
        idx_selected_train_prior = []
        for this_prior_class in priority_list:
            priority_tokens1 = set(unique_token_label_train[this_prior_class]) - set(unique_token_label_valid[this_prior_class])
            priority_tokens2 = set(unique_token_label_train[this_prior_class]) - set(unique_token_label_test[this_prior_class])
            priority_tokens = priority_tokens1.intersection(priority_tokens2)
            priority_tokens = list(priority_tokens)
            priority = {this_prior_class:priority_tokens}

        for label,index in label2idx_train.items():
            counter[index] = 0
        for this_token,this_y in zip(flat_tokens_train,y_train):
            this_class = idx2label_train[this_y]
            if this_class in priority:
                if this_y in priority[this_class] and counter[this_y] <= num_train:
                    idx_selected_train_prior.append(True)
                    counter[this_y] += 1
                else:
                    idx_selected_train_prior.append(False)
            else:
                idx_selected_train_prior.append(False)
    else:
        idx_selected_train_prior = [False]*len(flat_tokens_train)

    assert len(flat_tokens_train) == len(idx_selected_train_prior)

    for idx, (this_token,this_y) in enumerate(zip(flat_tokens_train,y_train)):
        if idx_selected_train_prior[idx]:
            idx_selected_train.append(False)
        elif idx2label_train[this_y] in special_classes:
            this_class = idx2label_train[this_y]
            if this_token in special_class_split['train'][this_class] and counter[this_y]<=num_train:
                idx_selected_train.append(True)
                counter[this_y] += 1
            else:
                idx_selected_train.append(False)
        elif counter[this_y]<=num_train:
            idx_selected_train.append(True)
            counter[this_y] += 1
        else:
            idx_selected_train.append(False)
    assert len(idx_selected_train) == len(flat_tokens_train)
    return idx_selected_train



def extract_sentence_attentions(
    sentence,
    model,
    tokenizer,
    device="cpu",
    aggregation="last",
    tokenization_counts={}
):
    """
    Adapt from https://neurox.qcri.org/docs/_modules/neurox/data/extraction/transformers_extractor.html#extract_sentence_representations
    """
    # this follows the HuggingFace API for transformers

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    original_tokens = sentence.split(" ")
    # Add a letter and space before each word since some tokenizers are space sensitive
    tmp_tokens = [
        "a" + " " + x if x_idx != 0 else x for x_idx, x in enumerate(original_tokens)
    ]
    assert len(original_tokens) == len(tmp_tokens)

    with torch.no_grad():
        # Get tokenization counts if not already available
        for token_idx, token in enumerate(tmp_tokens):
            tok_ids = [
                x for x in tokenizer.encode(token) if x not in special_tokens_ids
            ]
            if token_idx != 0:
                # Ignore the first token (added letter)
                tok_ids = tok_ids[1:]

            if token in tokenization_counts:
                assert tokenization_counts[token] == len(
                    tok_ids
                ), "Got different tokenization for already processed word"
            else:
                tokenization_counts[token] = len(tok_ids)
        ids = tokenizer.encode(sentence, truncation=True)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, num_heads, num_heads, sequence_length)
        # Tuple has 12 elements for base model: attention values at each layer
        all_attentions = model(input_ids)[-1]

        all_attentions = [
            attentions[0].cpu().numpy() for attentions in all_attentions
        ]
        # the expected shape is num_layer (12) x num_heads (12) x seq_len x seq_len
        all_attentions = np.array(all_attentions)


    # Remove special tokens
    ids_without_special_tokens = [x for x in ids if x not in special_tokens_ids]
    idx_without_special_tokens = [
        t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
    ]
    filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
    assert all_attentions.shape[2] == len(ids)
    assert all_attentions.shape[3] == len(ids)
    all_attentions = all_attentions[:, :, idx_without_special_tokens, :]
    all_attentions = all_attentions[:, :, :, idx_without_special_tokens]
    assert all_attentions.shape[2] == len(filtered_ids)
    assert all_attentions.shape[3] == len(filtered_ids)
    
    segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

    # Perform actual subword aggregation/detokenization
    counter_outer = 0
    detokenized_outer = []
    final_attentions = np.zeros(
        (all_attentions.shape[0],all_attentions.shape[1], len(original_tokens),len(original_tokens))
    )
    inputs_truncated = False
    total_len = sum(tokenization_counts.values())
    for _, token_outer in enumerate(tmp_tokens):
        current_word_start_idx_outer = counter_outer
        current_word_end_idx_outer = counter_outer + tokenization_counts[token_outer]
        detokenized_inner = []
        counter_inner = 0
        for _, token_inner in enumerate(tmp_tokens):
            current_word_start_idx_inner = counter_inner
            current_word_end_idx_inner = counter_inner + tokenization_counts[token_inner]

            # Check for truncated hidden states in the case where the
            # original word was actually tokenized
            if  (tokenization_counts[token] != 0 and current_word_start_idx_inner >= all_attentions.shape[2]) \
                    or current_word_end_idx_inner > all_attentions.shape[2]:
                final_attentions = final_attentions[:, :,:len(detokenized_outer),:len(detokenized_inner)]
                inputs_truncated = True
                print("You are here!!!!!!!!!!!!")
                break
            final_attentions[:, :,len(detokenized_outer),len(detokenized_inner)] = aggregate_repr(
                all_attentions,
                current_word_start_idx_outer,
                current_word_end_idx_outer - 1,
                current_word_start_idx_inner,
                current_word_end_idx_inner - 1,
                aggregation,
            )
            detokenized_inner.append(
                "".join(segmented_tokens[current_word_start_idx_inner:current_word_end_idx_inner])
            )
            counter_inner += tokenization_counts[token_inner]

        if inputs_truncated:
            break

        detokenized_outer.append(
                "".join(segmented_tokens[current_word_start_idx_outer:current_word_end_idx_outer])
            )
        counter_outer += tokenization_counts[token_outer]
    

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        assert counter_inner == len(ids_without_special_tokens)
        assert counter_outer == len(ids_without_special_tokens)
        assert len(detokenized_inner) == len(original_tokens)
        assert len(detokenized_outer) == len(original_tokens)
    print("===================================================================")

    return final_attentions


def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Adapt from https://neurox.qcri.org/docs/_modules/neurox/data/extraction/transformers_extractor.html#get_model_and_tokenizer
    """
    model_desc = model_desc.split(",")
    if len(model_desc) == 1:
        model_name = model_desc[0]
        tokenizer_name = model_desc[0]
    else:
        model_name = model_desc[0]
        tokenizer_name = model_desc[1]
    # https://huggingface.co/docs/transformers/v4.23.1/en/model_doc/bert#transformers.BertModel
    # model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start_x, end_x, start_y,end_y, aggregation):
    """
    Adapt from https://neurox.qcri.org/docs/_modules/neurox/data/extraction/transformers_extractor.html#aggregate_repr
    """
    if end_x < start_x or end_y< start_y:
        sys.stderr.write("WARNING: An empty slice of tokens was encountered. " +
            "This probably implies a special unicode character or text " +
            "encoding issue in your original data that was dropped by the " +
            "transformer model's tokenizer.\n")
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        assert 1 == 0, 'Not implemented yet'
    elif aggregation == "last":
        assert 1 == 0, 'Not implemented yet'
    elif aggregation == "average":
        temp = np.average(state[:, :, start_x : end_x + 1, start_y : end_y + 1], axis=2)
        output = np.average(temp[:, :,  : ], axis=2)
        return output


def extract_attentions(
    model_desc,
    input_corpus,
    device="cuda",
    aggregation="average",
    random_weights=False,
):
    """
    Adapt from https://neurox.qcri.org/docs/_modules/neurox/data/extraction/transformers_extractor.html#extract_representations
    """
    print(f"Loading model: {model_desc}")
    model, tokenizer = get_model_and_tokenizer(
        model_desc, device=device, random_weights=random_weights
    )

    print("Reading input corpus")

    def corpus_generator(input_corpus_path):
        with open(input_corpus_path, "r") as fp:
            for line in fp:
                yield line.strip()
            return

    output = {}
    print("Extracting attentions from model")
    tokenization_counts = {} # Cache for tokenizer rules
    for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
        attentions = extract_sentence_attentions(
            sentence,
            model,
            tokenizer,
            device=device,
            aggregation=aggregation,
            tokenization_counts=tokenization_counts
        )
        output[sentence_idx]=attentions.tolist()
    
    with open("attentions.json","w") as f:
        json.dump(output, f)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def EDA(flat_tokens_train,y_train,void_label):
    unique_token_label = {}
    for this_token, this_label in zip(flat_tokens_train,y_train):
        if this_label in unique_token_label:
            unique_token_label[this_label].add(this_token)
        else:
            unique_token_label[this_label] = set([this_token])
    for this_label in unique_token_label:
        print(f"label:{this_label}, the number of unique tokens:{len(unique_token_label[this_label])}")
        if this_label not in void_label:
            print(f"The unique labels are:{sorted(list(unique_token_label[this_label]))}")
    return unique_token_label