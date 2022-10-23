import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation
import neurox.data.control_task as ct
import neurox.interpretation.clustering
from sklearn.model_selection import train_test_split
import neurox.analysis.corpus as corpus
import numpy as np
import collections


def remove_seen_tokens(tokens,activations):
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
    transformers_extractor.extract_representations(model_description,
        file_in_name,
        activation_name,
        'cuda',
        aggregation="average" #last, first
    )


def load_extracted_activations(activation_file_name):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(activation_file_name,13)
    return activations


def load_tokens(activations,FILES_IN,FILES_LABEL):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens = data_loader.load_data(FILES_IN,
                                   FILES_LABEL,
                                   activations,
                                   512 # max_sent_length
                                  )
    return tokens


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
            this_weights = list(this_probe.parameters())[0].data.cpu().numpy()
            this_weights_mean = np.mean(np.abs(this_weights))
            # print(f"l1={this_l1},l2={this_l2}")
            # print("Absolute average value of parameters:",this_weights_mean)
            # print("Number of parameters that are not zero:",np.sum(this_weights != 0,axis=1))
            # print("Accuracy on the validation set:",this_score)
            if this_score['__OVERALL__'] > best_score:
                best_score = this_score['__OVERALL__']
                best_l1 = this_l1
                best_l2 = this_l2
                best_probe = this_probe
        return best_l1,best_l2,best_probe


def get_mappings(tokens,activations):
    ''' Get mappings for all models'''
    X, y, mapping = utils.create_tensors(tokens, activations, 'NAME') #mapping contains tuple of 4 dictionaries
    label2idx, idx2label, src2idx, idx2src = mapping

    return X, y, label2idx, idx2label, src2idx, idx2src


def all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,model_name):
    #Train the linear probes (logistic regression) - POS(code) tagging
    l1 = [0,0.001,0.01,0.1]
    l2 = [0,0.001,0.01,0.1]

    best_l1,best_l2,best_probe=param_tuning(X_train,y_train,X_valid,y_valid,idx2label,l1,l2)
    #Get scores of probes
    print(f"The best l1={best_l1}, the best l2={best_l2} for {model_name}")
    print(f"Accuracy on the test set of probing {model_name} of all layers:")
    scores = linear_probe.evaluate_probe(best_probe, X_test, y_test, idx_to_class=idx2label)
    print(scores)
    X_test_baseline = np.zeros_like(X_test)
    print(f"Accuracy on the test set of {model_name} model using the intercept:")
    linear_probe.evaluate_probe(best_probe, X_test_baseline, y_test, idx_to_class=idx2label)
    return best_probe, scores


def get_imp_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,probe,label2idx,idx2label,model_name):
    ''' Returns top 2% neurons for each model'''

    #Top neurons
    top_neurons, top_neurons_per_class = linear_probe.get_top_neurons(probe, 0.05, label2idx)
    print(f"{model_name} top neurons")
    print(repr(top_neurons))
    print(f"{model_name} top neurons per class")
    print(top_neurons_per_class)

    #Train probes on top neurons
    l1 = [0,0.001,0.01,0.1]
    l2 = [0,0.001,0.01,0.1]
    X_selected_train = ablation.filter_activations_keep_neurons(X_train, top_neurons)
    X_selected_valid = ablation.filter_activations_keep_neurons(X_valid, top_neurons)
    X_selected_test = ablation.filter_activations_keep_neurons(X_test, top_neurons)
    print("The shape of selected features",X_selected_train.shape)
    _,_,best_probe=param_tuning(X_selected_train,y_train,X_selected_valid,y_valid,idx2label,l1,l2)
    print(f"Accuracy on the test set of {model_name} model on top 5% neurons:")
    linear_probe.evaluate_probe(best_probe, X_selected_test, y_test, idx_to_class=idx2label)
    del X_selected_train,X_selected_valid,X_selected_test,best_probe

    ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx)
    X_selected_train = ablation.filter_activations_keep_neurons(X_train, ordering[:200])
    X_selected_valid = ablation.filter_activations_keep_neurons(X_valid, ordering[:200])
    X_selected_test = ablation.filter_activations_keep_neurons(X_test, ordering[:200])
    _,_,best_probe=param_tuning(X_selected_train,y_train,X_selected_valid,y_valid,idx2label,l1,l2)
    print(f"Accuracy on the test set of {model_name} model on top 200 neurons:")
    linear_probe.evaluate_probe(best_probe, X_selected_test, y_test, idx_to_class=idx2label)
    del X_selected_train,X_selected_valid,X_selected_test,best_probe
    return top_neurons


def get_top_words(top_neurons,tokens,activations,model_name):
    #relate neurons to corpus elements like words and sentences
    print(f"{model_name} top words")
    for neuron in top_neurons:
        top_words = corpus.get_top_words(tokens, activations, neuron, num_tokens=5)
        print(f"Top words for {model_name} neuron indx {neuron}",top_words)


def layerwise_probes_inference(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,model_name):
    ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''
    l1 = [0,0.001,0.01,0.1]
    l2 = [0,0.001,0.01,0.1]
    for i in range(13):
        print(f"{model_name} Layer", i)
        layer_train = ablation.filter_activations_by_layers(X_train, [i], 13)
        layer_valid = ablation.filter_activations_by_layers(X_valid, [i], 13)
        layer_test = ablation.filter_activations_by_layers(X_test, [i], 13)
        _,_,layer_probe=param_tuning(layer_train,y_train,layer_valid,y_valid,idx2label,l1,l2)
        del layer_train, layer_valid
        linear_probe.evaluate_probe(layer_probe, layer_test, y_test, idx_to_class=idx2label)
        del layer_test, layer_probe


def control_task_probes(X_train,y_train,X_test,y_test,idx2label_train,original_scores,model_name,method):
    print(f"Creating control dataset for {model_name} POS tagging task")
    label_freqs = collections.Counter(y_train)
    distribution = []
    if method == 'SAME':
        total = sum(label_freqs.values())
        for this_class,freq in label_freqs.items():
            distribution.append(freq/total)
    elif method == "UNIFORM":
        for this_class,freq in label_freqs.items():
            distribution.append(1/len(label_freqs))
    #random assign new class
    lookup_table = {}
    for this_class in label_freqs.keys():
        lookup_table[this_class] = np.random.choice(list(label_freqs.keys()), p=distribution)

    new_y_train = []
    new_y_test = []
    for this_y in y_train:
        new_this_y = lookup_table[this_y]
        new_y_train.append(new_this_y)
    for this_y in y_test:
        new_this_y = lookup_table[this_y]
        new_y_test.append(this_y)
    assert len(new_y_train) == len(y_train)
    assert len(new_y_test) == len(y_test)
    y_train = np.array(new_y_train)
    y_test = np.array(new_y_test)

    X_train, X_valid, y_train, y_valid = \
        train_test_split(X_train, y_train, test_size=0.1, shuffle=False)
    # normalization
    ct_norm = Normalization(X_train)
    X_train = ct_norm.norm(X_train)
    X_valid = ct_norm.norm(X_valid)
    X_test = ct_norm.norm(X_test)
    del ct_norm

    model_name = f'{model_name}_control_task'
    _, ct_scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,model_name)
    
    selectivity = original_scores['__OVERALL__'] - ct_scores['__OVERALL__']
    print(f'{model_name} Selectivity (Diff. between true task and probing task performance): ', selectivity)
    del ct_scores
    return selectivity


def probeless(X,y,model_name):
    '''General and Task specific probeless '''
    #Task specific : POS Tagging
    print(f"{model_name} Probless neuron ordering")
    print(neurox.interpretation.probeless.get_neuron_ordering(X,y))
    #General clustering analysis of POS dataset
    ''' Neuron Analysis on general tasks -- clustering'''
    print(f"{model_name} Clustering POS")
    print(neurox.interpretation.clustering.create_correlation_clusters(X, use_abs_correlation=True, clustering_threshold=0.5, method='average'))

def filter_by_frequency(X,y,label2idx,idx2label,threshold,model_name):
    count = collections.Counter(y)
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print(f"{model_name} distribution:")
    print(distribution)

    idx_selected = y <= threshold
    y = y[idx_selected]
    X = X[idx_selected]

    label2idx = {label:idx for (label,idx) in label2idx.items() if idx <= threshold}
    idx2label = {idx:label for (idx,label) in idx2label.items() if idx <= threshold}

    count = collections.Counter(y)
    distribution_rate = {k: v/len(y) for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print(f"{model_name} distribution after trauncating:")
    print(distribution_rate)
    print(distribution)
    print(label2idx)
    return X,y,label2idx,idx2label


def preprocess(activation_file_name,IN_file,LABEL_file,freq_threshold,model_name):
    activations = load_extracted_activations(activation_file_name)
    tokens =  load_tokens(activations,IN_file,LABEL_file)
    tokens,activations=remove_seen_tokens(tokens,activations)
    X, y, label2idx, idx2label, _, _ = get_mappings(tokens,activations)
    X_train, y_train, label2idx, idx2label = filter_by_frequency(X,y,label2idx,idx2label,freq_threshold,model_name)
    return X_train,y_train,label2idx,idx2label