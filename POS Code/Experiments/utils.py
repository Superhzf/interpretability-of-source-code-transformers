import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
# /work/LAS/cjquinn-lab/zefuh/selectivity/NeuroX_env/NeuroX/neurox/data/loader.py
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation
import neurox.data.control_task as ct
import neurox.interpretation.clustering
from sklearn.model_selection import train_test_split
import neurox.analysis.corpus as corpus
import numpy as np
import collections
import difflib
import torch
from transformers import AutoTokenizer, AutoModel

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
    transformers_extractor.extract_representations(model_description,
        file_in_name,
        activation_name,
        'cuda',
        aggregation="average" #last, first
    )


def load_extracted_activations(activation_file_name):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(activation_file_name)
    return activations


def load_tokens(activations,FILES_IN,FILES_LABEL):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens, sample_idx = data_loader.load_data(FILES_IN,
                                   FILES_LABEL,
                                   activations,
                                   512 # max_sent_length
                                  )
    return tokens, sample_idx


def param_tuning(X_train,y_train,X_valid,y_valid,idx2label,l1,l2,weight=None):
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
                                                                    batch_size=128,
                                                                    weight=weight)
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
    '''Re-organize the representation and labels such that they are ready for model training'''
    X, y, mapping = utils.create_tensors(tokens, activations, 'NAME') #mapping contains tuple of 4 dictionaries
    label2idx, idx2label, src2idx, idx2src = mapping

    return X, y, label2idx, idx2label, src2idx, idx2src


def all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,src_tokens_test,weighted,model_name,sample_idx_test=None):
    #Train the linear probes (logistic regression) - POS(code) tagging
    if weighted:
        classes = sorted(list(set(y_train)))
        count_classes = collections.Counter(y_train)
        total = sum(count_classes.values())
        weight = []
        for this_class in classes:
            this_weight =  count_classes[this_class]/total
            weight.append(this_weight)
        weight = torch.as_tensor(weight,device=torch.device('cuda'))
    else:
        weight = None
    best_l1,best_l2,best_probe=param_tuning(X_train,y_train,X_valid,y_valid,idx2label,l1,l2,weight)
    #Get scores of probes
    print()
    print(f"The best l1={best_l1}, the best l2={best_l2} for {model_name}")
    print(f"Accuracy on the test set of probing {model_name} of all layers:")
    scores,predictions = linear_probe.evaluate_probe(best_probe, X_test, y_test,idx_to_class=idx2label,
                                                    return_predictions=True,source_tokens=src_tokens_test)
    if src_tokens_test is not None:
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
        print(scores)
        print(f"Confusion matrix between NAME and KEYWORD:")
        print(f"NAME_NAME:{NAME_NAME},KW_NAME:{KW_NAME}")
        print(f"NAME_NAME_list:{NAME_NAME_list}")
        print(f"NAME_NAME_sample:{NAME_NAME_samples}",)
        print(f"NAME_KW:{NAME_KW},KW_KW:{KW_KW}")
        print(f"NAME_STRING:{NAME_STRING},KW_other:{KW_other}")
        print(f"NAME_NUMBER:{NAME_NUMBER}")
        print(f"NAME_STRING_list:{NAME_STRING_list}")
        print(f"NAME_NUMBER_list:{NAME_NUMBER_list}")
        print(f"NAME_NUMBER_sample:{NAME_NUMBER_samples}")
    X_test_baseline = np.zeros_like(X_test)
    print(f"Accuracy on the test set of {model_name} model using the intercept:")
    linear_probe.evaluate_probe(best_probe, X_test_baseline, y_test, idx_to_class=idx2label)
    return best_probe, scores


def get_imp_neurons(X_train,y_train,X_valid,y_valid,X_test,y_test,probe,label2idx,idx2label,src_tokens_test,weighted,model_name,sample_idx_test):
    ''' Returns top 2% neurons for each model'''

    #Top neurons
    # 0.05 means to select neurons that take top 5% mass
    top_neurons, top_neurons_per_class = linear_probe.get_top_neurons(probe, 0.05, label2idx)
    print()
    print(f"{model_name} top neurons")
    print(repr(top_neurons))
    print(f"{model_name} top neurons per class")
    print(top_neurons_per_class)

    #Train probes on top neurons
    X_selected_train = ablation.filter_activations_keep_neurons(X_train, top_neurons)
    X_selected_valid = ablation.filter_activations_keep_neurons(X_valid, top_neurons)
    X_selected_test = ablation.filter_activations_keep_neurons(X_test, top_neurons)
    print("The shape of selected features",X_selected_train.shape)
    this_model_name = f"{model_name}_top5%_mass"
    print("The shape of the training set:",X_selected_train.shape)
    print("The shape of the validation set:",X_selected_valid.shape)
    print("The shape of the testing set:",X_selected_test.shape)
    all_activations_probe(X_selected_train,y_train,X_selected_valid,y_valid,X_selected_test,y_test,idx2label,
                        src_tokens_test,weighted,this_model_name,sample_idx_test)

    ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx)
    X_selected_train = ablation.filter_activations_keep_neurons(X_train, ordering[:200])
    X_selected_valid = ablation.filter_activations_keep_neurons(X_valid, ordering[:200])
    X_selected_test = ablation.filter_activations_keep_neurons(X_test, ordering[:200])
    this_model_name = f"{model_name}_top200_neurons"
    all_activations_probe(X_selected_train,y_train,X_selected_valid,y_valid,X_selected_test,y_test,idx2label,
                        src_tokens_test,weighted,this_model_name,sample_idx_test)
    return top_neurons


def get_top_words(top_neurons,tokens,activations,model_name):
    #relate neurons to corpus elements like words and sentences
    """
    pretrained_BERT:
    idx: 5472, NAME
    [('decimal', 1.0), ('geos', 0.9418303459732197), ('tls', 0.9338263956575613), ('compat', 0.929471232136902), ('bson', 0.9242329985546578)]
    idx: 7008, STRING
    [('"KILLED"', 1.0), ('oslo', 0.9965087539321563), ('bson', 0.974736789295125), ('95', 0.9303350739266707), ('"SUCCEEDED"', 0.9185185340815739)]
    idx: 4704, NUMBER
    [('300', 1.0), ('69', 0.9377285482313947), ('1800', 0.9338789039515457), ('milliseconds', 0.9152723082592018), ('95', 0.9129744415244763)]
    idx: 8767,KEYWORD
    [('Else', 1.0), ('is', 0.8178607393924123), ('If', 0.8127052036231951), ('for', 0.7165972096789494), ('74.616338', 0.6740056796726646)]

    pretrained_CodeBERT:
    idx: 8794, NAME
    [('MULTILINE', 1.0), ('composite', 0.9702683773692685), ('subpath', 0.9541428350878021), ('None_', 0.9161060704527659), ('within', 0.9146583433499597)]
    idx: 2961, STRING
    [('Input', 1.0), ('"queue"', 0.9713504030157871), ('functional', 0.935347322951366), ('success', 0.8734251315332126), ('"time"', 0.87176689098753)]
    idx: 6245, NUMBER
    [('35', 1.0), ('42', 0.8939844114777427), ('closing', 0.88509228271514), ('sites', 0.8689794668738634), ('21', 0.819644843478075)]
    idx:6638, KEYWORD
    [('for', 1.0), ('else', 0.8821944740098403), ('except', 0.7865639726358226), ('while', 0.7608804667886525), ('with', 0.7143995338083216)]

    pretrained_GraphCodeBERT
    idx:1037, NAME
    [('horizon', 1.0), ('WAYS', 0.9671729225278997), ('hyper', 0.9104408868432127), ('pyramid', 0.8298599115554878), ('LOG', 0.8088905754130504)]
    idx: 6205, STRING
    [('District', 1.0), ('fabric', 0.9752651660457131), ('is', 0.9630007074476946), ('"store_true"', 0.9231785144825696), ('pushkin', 0.9162305723018496)]
    idx:2043 , NUMBER
    [('next', 1.0), ('2014', 0.7701208102064382), ('reflection', 0.7159774590936444), ('current', 0.6914945816893433), ('Context', 0.625575550273334)]
    idx:6638 ,KEYWORD
    [('WAY', 1.0), ('else', 0.9172237419246152), ('or', 0.9137186089590944), ('for', 0.8729917810680667), ('GetBoard', 0.867617197937286)]
    """
    print(f"{model_name} top words")
    for neuron in top_neurons:
        top_words = corpus.get_top_words(tokens, activations, neuron, num_tokens=5)
        print(f"Top words for {model_name} neuron indx {neuron}",top_words)


def layerwise_probes_inference(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label,src_tokens_test,weighted,model_name,sample_idx_test):
    ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''
    for i in range(13):
        print(f"{model_name} Layer", i)
        this_model_name = f"{model_name}_layer_{i}"
        layer_train = ablation.filter_activations_by_layers(X_train, [i], 13)
        layer_valid = ablation.filter_activations_by_layers(X_valid, [i], 13)
        layer_test = ablation.filter_activations_by_layers(X_test, [i], 13)
        _,_ = all_activations_probe(layer_train,y_train,layer_valid,y_valid,layer_test,y_test,
                                    idx2label,src_tokens_test,weighted,this_model_name,sample_idx_test)


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
    _, ct_scores = all_activations_probe(X_train,y_train,X_valid,y_valid,X_test,y_test,idx2label_train,None,False,model_name, None)
    
    selectivity = original_scores['__OVERALL__'] - ct_scores['__OVERALL__']
    print()
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


def filterByClass(tokens,activations,X,y,label2idx,model_name,sample_idx):
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

    class_wanted = ['NAME','STRING','NUMBER','KEYWORD']
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


def preprocess(activation_file_name,IN_file,LABEL_file,remove_seen_tokens,model_name):
    activations = load_extracted_activations(activation_file_name)
    tokens, sample_idx =  load_tokens(activations,IN_file,LABEL_file)
    if remove_seen_tokens:
        tokens,activations=removeSeenTokens(tokens,activations)
    X, y, label2idx, _, _, _ = get_mappings(tokens,activations)
    tokens,activations,flat_src_tokens,X_train, y_train, label2idx, idx2label, sample_idx = filterByClass(tokens,activations,X,y,label2idx,model_name,sample_idx)
    return tokens,activations,flat_src_tokens,X_train,y_train,label2idx,idx2label, sample_idx


def selectBasedOnTrain(flat_tokens_test,X_test, y_test,flat_tokens_train,label2idx_train,keyword_list_test,sample_idx_test=None):
    idx_selected = []
    count_number = 0
    count_name = 0
    count_keyword = 0
    count_str = 0
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
                    if count_str>=280 or getOverlap(this_token_test,this_token_train) >= 4:
                        is_selected = False
                        break
                if is_selected:
                    count_str += 1
            elif this_y_test == label2idx_train['NUMBER']:
                for this_token_train in flat_tokens_train:
                    if count_number>=280 or getOverlap(this_token_test,this_token_train) >= 3:
                        is_selected = False
                        break
                if is_selected:
                    count_number += 1
            elif this_y_test == label2idx_train['NAME']:
                for this_token_train in flat_tokens_train:
                    if count_name>= 280 or getOverlap(this_token_test,this_token_train) >= 2:
                        is_selected = False
                        break
                if is_selected:
                    count_name += 1
            elif this_y_test == label2idx_train['KEYWORD']:
                if this_token_test not in keyword_list_test or count_keyword >= 280:
                    is_selected = False
                else:
                    count_keyword += 1
            idx_selected.append(is_selected)
    assert len(idx_selected) == len(flat_tokens_test)
    flat_tokens_test = flat_tokens_test[idx_selected]
    X_test = X_test[idx_selected]
    y_test = y_test[idx_selected]
    if sample_idx_test is not None:
        sample_idx_test = sample_idx_test[idx_selected]
        assert len(sample_idx_test) == len(y_test)
    return X_test, y_test, flat_tokens_test, idx_selected, sample_idx_test


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
    counter = 0
    detokenized = []
    final_attentions = np.zeros(
        (all_attentions.shape[0],all_attentions.shape[1], len(original_tokens),len(original_tokens))
    )
    inputs_truncated = False

    for token_idx, token in enumerate(tmp_tokens):
        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]

        # Check for truncated hidden states in the case where the
        # original word was actually tokenized
        if  (tokenization_counts[token] != 0 and current_word_start_idx >= all_attentions.shape[2]) \
                or current_word_end_idx > all_attentions.shape[2]:
            final_attentions = final_attentions[:, :,:len(detokenized),:len(detokenized)]
            inputs_truncated = True
            break
        final_attentions[:, :,len(detokenized),len(detokenized)] = aggregate_repr(
            all_attentions,
            current_word_start_idx,
            current_word_end_idx - 1,
            aggregation,
        )
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )
        counter += tokenization_counts[token]
        
    print(final_attentions)
    exit(0)

    print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
    print("Counter: %d" % (counter))

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        assert counter == len(ids_without_special_tokens)
        assert len(detokenized) == len(original_tokens)
    print("===================================================================")

    return final_attentions, detokenized


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
    model = AutoModel.from_pretrained(model_name, output_attentions=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, aggregation):
    """
    Adapt from https://neurox.qcri.org/docs/_modules/neurox/data/extraction/transformers_extractor.html#aggregate_repr
    """
    if end < start:
        sys.stderr.write("WARNING: An empty slice of tokens was encountered. " +
            "This probably implies a special unicode character or text " +
            "encoding issue in your original data that was dropped by the " +
            "transformer model's tokenizer.\n")
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        return state[:, :, start, start]
    elif aggregation == "last":
        return state[:, :, end, end]
    elif aggregation == "average":
        temp = np.average(state[:, :, start : end + 1, start : end + 1], axis=2)
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


    print("Extracting representations from model")
    tokenization_counts = {} # Cache for tokenizer rules
    for sentence_idx, sentence in enumerate(corpus_generator(input_corpus)):
        attentions, extracted_words = extract_sentence_attentions(
            sentence,
            model,
            tokenizer,
            device=device,
            aggregation=aggregation,
            tokenization_counts=tokenization_counts
        )
    
        print(f"The idx of this line of code:{sentence_idx}")
        print(f"Shape of the attention: {attentions.shape}")
        print(attentions[0:3,0:3,:,:])