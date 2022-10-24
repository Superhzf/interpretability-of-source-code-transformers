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
    this_model_name = f"{model_name}_top5%_neurons"
    print("The shape of the training set:",X_train.shape)
    print("The shape of the validation set:",X_valid.shape)
    print("The shape of the testing set:",X_test.shape)
    all_activations_probe(X_selected_train,y_train,X_selected_valid,y_valid,X_selected_test,y_test,idx2label,this_model_name)

    ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx)
    X_selected_train = ablation.filter_activations_keep_neurons(X_train, ordering[:200])
    X_selected_valid = ablation.filter_activations_keep_neurons(X_valid, ordering[:200])
    X_selected_test = ablation.filter_activations_keep_neurons(X_test, ordering[:200])
    this_model_name = f"{model_name}_top200_neurons"
    all_activations_probe(X_selected_train,y_train,X_selected_valid,y_valid,X_selected_test,y_test,idx2label,this_model_name)
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
        this_model_name = f"{model_name}_layer_{i}"
        layer_train = ablation.filter_activations_by_layers(X_train, [i], 13)
        layer_valid = ablation.filter_activations_by_layers(X_valid, [i], 13)
        layer_test = ablation.filter_activations_by_layers(X_test, [i], 13)
        _,_ = all_activations_probe(layer_train,y_train,layer_valid,y_valid,layer_test,y_test,idx2label,this_model_name)


def randomReassignment(tokens,labels,distribution):
    lookup_table={}
    #random assign new class
    # for this_class in label_freqs.keys():
    #     lookup_table[this_class] = np.random.choice(list(label_freqs.keys()), p=distribution)

    for this_token in tokens:
            if this_token not in lookup_table:
                lookup_table[this_token] = np.random.choice(labels, p=distribution)
    y_ct = []
    for this_token in tokens:
        this_y_ct = lookup_table[this_token]
        y_ct.append(this_y_ct)
    return y_ct


def control_task_probes(tokens_train,X_train,y_train,tokens_test,X_test,y_test,idx2label_train,original_scores,model_name,method):
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
    else:
        assert 1==0, "method is not understood"
    while True:
        y_train_ct = randomReassignment(tokens_train,list(label_freqs.keys()),distribution)
        y_test_ct = randomReassignment(tokens_test,list(label_freqs.keys()),distribution)
        assert len(y_train_ct) == len(y_train)
        assert len(y_test_ct) == len(y_test)
        y_train_ct = np.array(y_train_ct)
        y_test_ct = np.array(y_test_ct)

        X_train_ct, X_valid_ct, y_train_ct, y_valid_ct = \
            train_test_split(X_train, y_train_ct, test_size=0.1, shuffle=False)
        # class 0,1,2 must be in y_train_ct
        if 0 in y_train_ct and 1 in y_train_ct and 2 in y_train_ct:
            break
    y_train = y_train_ct
    y_valid = y_valid_ct
    y_test = y_test_ct
    del y_train_ct,y_valid_ct,y_test_ct
    # normalization
    ct_norm = Normalization(X_train_ct)
    X_train = ct_norm.norm(X_train_ct)
    X_valid = ct_norm.norm(X_valid_ct)
    X_test = ct_norm.norm(X_test)
    del ct_norm

    assert X_train.shape[0] == len(y_train)
    assert X_valid.shape[0] == len(y_valid)
    assert X_test.shape[0] == len(y_test)
    model_name = f'{model_name}_control_task'
    _, ct_scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label_train,model_name)
    
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


def alignTokenAct(tokens,activations,idx_selected):
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
            this_new_activations.append(this_new_activations)
            new_tokens_src.append(this_new_tokens_src)
            new_tokens_trg.append(this_new_tokens_trg)
            new_activations.append(this_new_activations)
    assert idx == len(idx_selected)
    new_tokens = {'source':new_tokens_src,'target':new_tokens_trg}
    return new_tokens,new_activations


def filter_by_frequency(tokens,activations,X,y,label2idx,idx2label,threshold,model_name):
    count = collections.Counter(y)
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
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
    assert len(np.array([l for sublist in activations for l in sublist])) == len(flat_src_tokens)
    assert len(np.array([l for sublist in tokens['target'] for l in sublist])) == len(flat_src_tokens)

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


def preprocess(activation_file_name,IN_file,LABEL_file,freq_threshold,model_name):
    activations = load_extracted_activations(activation_file_name)
    tokens =  load_tokens(activations,IN_file,LABEL_file)
    tokens,activations=remove_seen_tokens(tokens,activations)
    X, y, label2idx, idx2label, _, _ = get_mappings(tokens,activations)
    tokens,activations,flat_src_tokens,X_train, y_train, label2idx, idx2label = filter_by_frequency(tokens,activations,X,y,label2idx,idx2label,freq_threshold,model_name)
    return tokens,activations,flat_src_tokens,X_train,y_train,label2idx,idx2label
