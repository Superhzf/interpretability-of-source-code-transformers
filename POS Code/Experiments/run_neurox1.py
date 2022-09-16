"""
Created on Tue Apr 12 14:21:21 2022

@author: sharm
"""
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation
import neurox.data.control_task as ct
import neurox.interpretation.clustering
import neurox.interpretation.probeless
from sklearn.model_selection import train_test_split
import neurox.analysis.corpus as corpus
import numpy as np


class Normalization:
    def __init__(self,df):
        self.var_mean = np.mean(df,axis=0)
        self.var_std = np.std(df,axis=0)

    def norm(self,df):
        norm_df = (df-self.var_mean)/self.var_std
        return norm_df


def preprocessing():
    ''' Create codetest.in and codetest.label from python code files'''

    def skiplines():
        #Modify the cloned NeuroX library to remove the restriction on number of skip lines. Consolidate everything.Include preprocessing here.
        None


#Extract activations.json files
def extract_activations():
    #Extract representations from BERT
    transformers_extractor.extract_representations('bert-base-uncased',
        'codetest2_unique.in',
        'bert_activations.json',
        'cuda',
        aggregation="average" #last, first
    )

    #Extract representations from CodeBERT
    transformers_extractor.extract_representations('microsoft/codebert-base',
        'codetest2_unique.in',
        'codebert_activations.json',
        'cuda',
        aggregation="average" #last, first
    )

    #Extract representations from GraphCodeBERT
    transformers_extractor.extract_representations('microsoft/graphcodebert-base',
        'codetest2_unique.in',
        'graphcodebert_activations.json',
        'cuda',
        aggregation="average" #last, first
    )

    return(load_extracted_activations())


def load_extracted_activations():
    #Load activations from json files
    bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
    codebert_activations, codebert_num_layers = data_loader.load_activations('codebert_activations.json',13) #num_layers is 13 not 768
    graphcodebert_activations, graphcodebert_num_layers = data_loader.load_activations('graphcodebert_activations.json',13)

    return bert_activations, codebert_activations, graphcodebert_activations

def load_tokens(bert_activations,codebert_activations, graphcodebert_activations):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    bert_tokens = data_loader.load_data('codetest2_unique.in',
                                   'codetest2_unique.label',
                                   bert_activations,
                                   512 # max_sent_length
                                  )

    codebert_tokens = data_loader.load_data('codetest2_unique.in',
                                   'codetest2_unique.label',
                                   codebert_activations,
                                   512 # max_sent_length
                                  )

    graphcodebert_tokens = data_loader.load_data('codetest2_unique.in',
                                   'codetest2_unique.label',
                                   graphcodebert_activations,
                                   512 # max_sent_length
                                  )


    return bert_tokens, codebert_tokens, graphcodebert_tokens



def linear_probes_inference( bert_tokens, bert_activations, codebert_tokens, codebert_activations, graphcodebert_tokens, graphcodebert_activations):
    ''' Returns models and accuracy(score) of the probes trained on entire activation space '''

    def get_mappings():
        ''' Get mappings for all models'''
        bert_X, bert_y, bert_mapping = utils.create_tensors(bert_tokens, bert_activations, 'NAME') #mapping contains tuple of 4 dictionaries
        bert_label2idx, bert_idx2label, bert_src2idx, bert_idx2src = bert_mapping

        codebert_X, codebert_y, codebert_mapping = utils.create_tensors(codebert_tokens, codebert_activations, 'NAME') #mapping contains tuple of 4 dictionaries
        codebert_label2idx, codebert_idx2label, codebert_src2idx, codebert_idx2src = codebert_mapping

        graphcodebert_X, graphcodebert_y, graphcodebert_mapping = utils.create_tensors(graphcodebert_tokens, graphcodebert_activations, 'NAME') #mapping contains tuple of 4 dictionaries
        graphcodebert_label2idx, graphcodebert_idx2label, graphcodebert_src2idx, graphcodebert_idx2src = graphcodebert_mapping

        return bert_X, bert_y, codebert_X, codebert_y,  bert_label2idx, bert_idx2label, bert_src2idx, bert_idx2src, codebert_label2idx, codebert_idx2label, codebert_src2idx, codebert_idx2src, graphcodebert_X, graphcodebert_y, graphcodebert_label2idx, graphcodebert_idx2label, graphcodebert_src2idx, graphcodebert_idx2src

    def all_activations_probe():
        #Train the linear probes (logistic regression) - POS(code) tagging

        bert_probe = linear_probe.train_logistic_regression_probe(bert_X_train, bert_y_train, lambda_l1=0.001, lambda_l2=0.001)
        codebert_probe = linear_probe.train_logistic_regression_probe(codebert_X_train, codebert_y_train, lambda_l1=0.001, lambda_l2=0.001)
        graphcodebert_probe = linear_probe.train_logistic_regression_probe(graphcodebert_X_train, graphcodebert_y_train, lambda_l1=0.001, lambda_l2=0.001)

        #Evaluate linear probes for POS(code) tagging
        linear_probe.evaluate_probe(bert_probe, bert_X_test, bert_y_test, idx_to_class=bert_idx2label)
        linear_probe.evaluate_probe(codebert_probe, codebert_X_test, codebert_y_test, idx_to_class=codebert_idx2label)
        linear_probe.evaluate_probe(graphcodebert_probe, graphcodebert_X_test, graphcodebert_y_test, idx_to_class=graphcodebert_idx2label)

        #Get scores of probes
        bert_scores = linear_probe.evaluate_probe(bert_probe, bert_X_test, bert_y_test, idx_to_class=bert_idx2label)
        print(bert_scores)
        codebert_scores = linear_probe.evaluate_probe(codebert_probe, codebert_X_test, codebert_y_test, idx_to_class=codebert_idx2label)
        print(codebert_scores)
        graphcodebert_scores = linear_probe.evaluate_probe(graphcodebert_probe, graphcodebert_X_test, graphcodebert_y_test, idx_to_class=graphcodebert_idx2label)
        print(graphcodebert_scores)
        return bert_probe, codebert_probe, graphcodebert_probe, bert_scores, codebert_scores, graphcodebert_scores


    def get_imp_neurons():
        ''' Returns top 10% neurons for each model'''

        #Get neuron ordering
        bert_ordering, bert_cutoffs = linear_probe.get_neuron_ordering(bert_probe, bert_label2idx)
        codebert_ordering, codebert_cutoffs = linear_probe.get_neuron_ordering(codebert_probe, codebert_label2idx)
        graphcodebert_ordering, graphcodebert_cutoffs = linear_probe.get_neuron_ordering(graphcodebert_probe, graphcodebert_label2idx)

        #Top neurons
        bert_top_neurons, bert_top_neurons_per_class = linear_probe.get_top_neurons(bert_probe, 0.02, bert_label2idx)
        print("Bert top neurons")
        print(repr(bert_top_neurons))
        print("Bert top neurons per class")
        print(bert_top_neurons_per_class)

        codebert_top_neurons, codebert_top_neurons_per_class = linear_probe.get_top_neurons(codebert_probe, 0.02, codebert_label2idx)
        print("CodeBert top neurons")
        print(repr(codebert_top_neurons))
        print("CodeBert top neurons per class")
        print(codebert_top_neurons_per_class)

        graphcodebert_top_neurons, graphcodebert_top_neurons_per_class = linear_probe.get_top_neurons(graphcodebert_probe, 0.02, graphcodebert_label2idx)
        print("GraphCodeBert top neurons")
        print(repr(graphcodebert_top_neurons))
        print("GraphCodeBert top neurons per class")
        print(graphcodebert_top_neurons_per_class)

        #Train probes on top neurons and save them
        bert_X_selected = ablation.filter_activations_keep_neurons(bert_X_train, bert_ordering[:100])
        bert_X_selected.shape
        bert_probe_selected = linear_probe.train_logistic_regression_probe(bert_X_selected, bert_y_train, lambda_l1=0.001, lambda_l2=0.001)
        pickle.dump(bert_probe_selected, open("bert_probe_selected.sav", 'wb'))
        del bert_X_selected
        bert_X_selected_test = ablation.filter_activations_keep_neurons(bert_X_test, bert_ordering[:100])
        linear_probe.evaluate_probe(bert_probe_selected, bert_X_selected_test, bert_y_test, idx_to_class=bert_idx2label)
        del bert_X_selected_test

        codebert_X_selected = ablation.filter_activations_keep_neurons(codebert_X_train, codebert_ordering[:100])
        codebert_X_selected.shape
        codebert_probe_selected = linear_probe.train_logistic_regression_probe(codebert_X_selected, codebert_y_train, lambda_l1=0.001, lambda_l2=0.001)
        pickle.dump(codebert_probe_selected, open("codebert_probe_selected.sav", 'wb'))
        del codebert_X_selected
        codebert_X_selected_test = ablation.filter_activations_keep_neurons(codebert_X_test, codebert_ordering[:100])
        linear_probe.evaluate_probe(codebert_probe_selected, codebert_X_selected_test, codebert_y_test, idx_to_class=codebert_idx2label)
        del codebert_X_selected_test

        graphcodebert_X_selected = ablation.filter_activations_keep_neurons(graphcodebert_X_train, graphcodebert_ordering[:100])
        graphcodebert_X_selected.shape
        graphcodebert_probe_selected = linear_probe.train_logistic_regression_probe(graphcodebert_X_selected, graphcodebert_y_train, lambda_l1=0.001, lambda_l2=0.001)
        del graphcodebert_X_selected
        pickle.dump(graphcodebert_probe_selected, open("graphcodebert_probe_selected.sav", 'wb'))
        graphcodebert_X_selected_test = ablation.filter_activations_keep_neurons(graphcodebert_X_test, graphcodebert_ordering[:100])
        linear_probe.evaluate_probe(graphcodebert_probe_selected, graphcodebert_X_selected_test, graphcodebert_y_test, idx_to_class=graphcodebert_idx2label)
        del graphcodebert_X_selected_test

        return bert_top_neurons, codebert_top_neurons, graphcodebert_top_neurons

    def get_top_words(bert_top_neurons, codebert_top_neurons, graphcodebert_top_neurons):
        #relate neurons to corpus elements like words and sentences

        print("BERT top words")
        for neuron in bert_top_neurons:
            bert_top_words = corpus.get_top_words(bert_tokens, bert_activations, neuron, num_tokens=5)
            print(f"Top words for bert neuron indx {neuron}",bert_top_words)
        print("----------------------------------------------------------------")
        print("CodeBERT top words")
        for neuron in codebert_top_neurons:
            codebert_top_words = corpus.get_top_words(codebert_tokens, codebert_activations, neuron, num_tokens=5)
            print(f"Top words for codebert neuron indx {neuron}",codebert_top_words)
        print("----------------------------------------------------------------")
        print("GraphCodeBERT top words")
        for neuron in graphcodebert_top_neurons:
            graphcodebert_top_words = corpus.get_top_words(graphcodebert_tokens, graphcodebert_activations, neuron, num_tokens=5)
            print(f"Top words for graphcodebert neuron indx {neuron}",graphcodebert_top_words)


    def layerwise_probes_inference():
        ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''

        #BERT
        for i in range(13):
            print("Bert Layer", i)
            layer_train = ablation.filter_activations_by_layers(bert_X_train, [i], 13)
            layer_probe = linear_probe.train_logistic_regression_probe(layer_train, bert_y_train, lambda_l1=0.001, lambda_l2=0.001)
            del layer_train
            pickle.dump(layer_probe, open(f"bert_layer{i}_probe.sav", 'wb'))
            layer_test = ablation.filter_activations_by_layers(bert_X_test, [i], 13)
            linear_probe.evaluate_probe(layer_probe, layer_test, bert_y_test, idx_to_class=bert_idx2label)
            del layer_test
            del layer_probe

        #CodeBERT
        for i in range(13):
            print("Codebert Layer", i)
            layer_train = ablation.filter_activations_by_layers(codebert_X_train, [i], 13)
            layer_probe = linear_probe.train_logistic_regression_probe(layer_train, codebert_y_train, lambda_l1=0.001, lambda_l2=0.001)
            del layer_train
            pickle.dump(layer_probe, open(f"codebert_layer{i}_probe.sav", 'wb'))
            layer_test = ablation.filter_activations_by_layers(codebert_X_test, [i], 13)
            linear_probe.evaluate_probe(layer_probe, layer_test, codebert_y_test, idx_to_class=codebert_idx2label)
            del layer_test
            del layer_probe

         #GraphCodeBERT
        for i in range(13):
            print("GraphCodebert Layer", i)
            layer_train = ablation.filter_activations_by_layers(graphcodebert_X_train, [i], 13)
            layer_probe = linear_probe.train_logistic_regression_probe(layer_train, graphcodebert_y_train, lambda_l1=0.001, lambda_l2=0.001)
            del layer_train
            pickle.dump(layer_probe, open(f"graphcodebert_layer{i}_probe.sav", 'wb'))
            layer_test = ablation.filter_activations_by_layers(graphcodebert_X_test, [i], 13)
            linear_probe.evaluate_probe(layer_probe, layer_test, graphcodebert_y_test, idx_to_class=graphcodebert_idx2label)
            del layer_test
            del layer_probe

    def control_task_probes(bert_scores, codebert_scores, graphcodebert_scores):
        print("Creating control dataset for BERT POS tagging task")
        [bert_ct_tokens] = ct.create_sequence_labeling_dataset(bert_tokens, sample_from='uniform')
        print([s+'/'+str(t) for s,t in zip(bert_ct_tokens['source'][0], bert_ct_tokens['target'][0])])
        bert_X_ct, bert_y_ct, bert_mapping_ct = utils.create_tensors(bert_ct_tokens, bert_activations, 'NAME')
        bert_label2idx_ct, bert_idx2label_ct, bert_src2idx_ct, bert_idx2src_ct = bert_mapping_ct

        bert_X_ct_train, bert_X_ct_test, bert_y_ct_train, bert_y_ct_test = \
            train_test_split(bert_X_ct, bert_y_ct, test_size=0.2,random_state=50, shuffle=False)
        # normalization
        bert_ct_norm = Normalization(bert_X_ct_train)
        bert_X_ct_train = bert_ct_norm.norm(bert_X_ct_train)
        bert_X_ct_test = bert_ct_norm.norm(bert_X_ct_test)
        del bert_ct_norm

        bert_ct_probe = linear_probe.train_logistic_regression_probe(bert_X_ct_train, bert_y_ct_train, lambda_l1=0.001, lambda_l2=0.001)
        bert_ct_scores = linear_probe.evaluate_probe(bert_ct_probe, bert_X_ct_test, bert_y_ct_test, idx_to_class=bert_idx2label_ct)
        bert_selectivity = bert_scores['__OVERALL__'] - bert_ct_scores['__OVERALL__']
        print('BERT Selectivity (Diff. between true task and probing task performance): ', bert_selectivity)
        del bert_ct_probe
        del bert_ct_scores
        del bert_X_ct_train, bert_y_ct_train, bert_X_ct_test, bert_y_ct_test

        print("Creating control dataset for CodeBERT POS tagging task")
        [codebert_ct_tokens] = ct.create_sequence_labeling_dataset(codebert_tokens, sample_from='uniform')
        print([s+'/'+str(t) for s,t in zip(codebert_ct_tokens['source'][0], codebert_ct_tokens['target'][0])])
        codebert_X_ct, codebert_y_ct, codebert_mapping_ct = utils.create_tensors(codebert_ct_tokens, codebert_activations, 'NAME')
        codebert_label2idx_ct, codebert_idx2label_ct, codebert_src2idx_ct, codebert_idx2src_ct = codebert_mapping_ct

        codebert_X_ct_train, codebert_X_ct_test, codebert_y_ct_train, codebert_y_ct_test = \
            train_test_split(codebert_X_ct, codebert_y_ct, test_size=0.2,random_state=50, shuffle=False)
        # normalization
        codebert_ct_norm = Normalization(codebert_X_ct_train)
        codebert_X_ct_train = codebert_ct_norm.norm(codebert_X_ct_train)
        codebert_X_ct_test = codebert_ct_norm.norm(codebert_X_ct_test)
        del codebert_ct_norm

        codebert_ct_probe = linear_probe.train_logistic_regression_probe(codebert_X_ct_train, codebert_y_ct_train, lambda_l1=0, lambda_l2=0)
        codebert_ct_scores = linear_probe.evaluate_probe(codebert_ct_probe, codebert_X_ct_test, codebert_y_ct_test, idx_to_class=codebert_idx2label_ct)
        codebert_selectivity = codebert_scores['__OVERALL__'] - codebert_ct_scores['__OVERALL__']
        print('CodeBERT Selectivity (Diff. between true task and probing task performance): ', codebert_selectivity)
        del codebert_ct_probe
        del codebert_ct_scores
        del codebert_X_ct_train, codebert_y_ct_train, codebert_X_ct_test, codebert_y_ct_test

        print("Creating control dataset for GraphCodeBERT POS tagging task")
        [graphcodebert_ct_tokens] = ct.create_sequence_labeling_dataset(graphcodebert_tokens, sample_from='uniform')
        print([s+'/'+str(t) for s,t in zip(graphcodebert_ct_tokens['source'][0], graphcodebert_ct_tokens['target'][0])])
        graphcodebert_X_ct, graphcodebert_y_ct, graphcodebert_mapping_ct = utils.create_tensors(graphcodebert_ct_tokens, graphcodebert_activations, 'NAME')
        graphcodebert_label2idx_ct, graphcodebert_idx2label_ct, graphcodebert_src2idx_ct, graphcodebert_idx2src_ct = graphcodebert_mapping_ct

        graphcodebert_X_ct_train, graphcodebert_X_ct_test, graphcodebert_y_ct_train, graphcodebert_y_ct_test = \
            train_test_split(codebert_X_ct, codebert_y_ct, test_size=0.2,random_state=50, shuffle=False)
        # normalization
        graphcodebert_ct_norm = Normalization(graphcodebert_X_ct_train)
        graphcodebert_X_ct_train = graphcodebert_ct_norm.norm(graphcodebert_X_ct_train)
        graphcodebert_X_ct_test = graphcodebert_ct_norm.norm(graphcodebert_X_ct_test)
        del graphcodebert_ct_norm

        graphcodebert_ct_probe = linear_probe.train_logistic_regression_probe(graphcodebert_X_ct_train, graphcodebert_y_ct_train, lambda_l1=0.001, lambda_l2=0.001)
        graphcodebert_ct_scores = linear_probe.evaluate_probe(graphcodebert_ct_probe, graphcodebert_X_ct_test, graphcodebert_y_ct_test, idx_to_class=graphcodebert_idx2label_ct)
        graphcodebert_selectivity = graphcodebert_scores['__OVERALL__'] - graphcodebert_ct_scores['__OVERALL__']
        print('GraphCodeBERT Selectivity (Diff. between true task and probing task performance): ', graphcodebert_selectivity)
        del graphcodebert_ct_probe
        del graphcodebert_ct_scores
        del graphcodebert_X_ct_train, graphcodebert_y_ct_train, graphcodebert_X_ct_test, graphcodebert_y_ct_test

        return bert_selectivity, codebert_selectivity, graphcodebert_selectivity

    def probeless(bert_X,bert_y, codebert_X, codebert_y, graphcodebert_X, graphcodebert_y):
        '''General and Task specific probeless '''
        #Task specific : POS Tagging
        print("BERT Probless neuron ordering")
        print(neurox.interpretation.probeless.get_neuron_ordering(bert_X,bert_y))
        print("CodeBERT Probless neuron ordering")
        print(neurox.interpretation.probeless.get_neuron_ordering(codebert_X,codebert_y))
        print("GraphCodeBERT Probless neuron ordering")
        print(neurox.interpretation.probeless.get_neuron_ordering(graphcodebert_X,graphcodebert_y))

        #General clustering analysis of POS dataset
        ''' Neuron Analysis on general tasks -- clustering'''
        print("BERT Clustering POS")
        print(neurox.interpretation.clustering.create_correlation_clusters(bert_X, use_abs_correlation=True, clustering_threshold=0.5, method='average'))
        print("CodeBERT Clustering POS")
        print(neurox.interpretation.clustering.create_correlation_clusters(codebert_X, use_abs_correlation=True, clustering_threshold=0.5, method='average'))
        print("GraphCodeBERT Clustering POS")
        print(neurox.interpretation.clustering.create_correlation_clusters(graphcodebert_X, use_abs_correlation=True, clustering_threshold=0.5, method='average'))


    #Get mappings
    bert_X, bert_y, codebert_X, codebert_y,  bert_label2idx, bert_idx2label, \
    bert_src2idx, bert_idx2src, codebert_label2idx, codebert_idx2label, \
    codebert_src2idx, codebert_idx2src, graphcodebert_X, graphcodebert_y, \
    graphcodebert_label2idx, graphcodebert_idx2label, graphcodebert_src2idx, \
    graphcodebert_idx2src = get_mappings()

    import collections
    count = collections.Counter(bert_y)
    distribution = {k: v for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
    print("distribution:")
    print(distribution)

    print("bert_label2idx")
    print(bert_label2idx)

    exit(0)
    idx_selected = bert_y <= 43
    bert_y = bert_y[idx_selected]
    bert_X = bert_X[idx_selected]

    codebert_y = codebert_y[idx_selected]
    codebert_X = codebert_X[idx_selected]

    graphcodebert_y = graphcodebert_y[idx_selected]
    graphcodebert_X = graphcodebert_X[idx_selected]


    bert_X_train, bert_X_test, bert_y_train, bert_y_test = \
        train_test_split(bert_X, bert_y, test_size=0.2,random_state=50, shuffle=False)
    codebert_X_train, codebert_X_test, codebert_y_train, codebert_y_test = \
        train_test_split(codebert_X, codebert_y, test_size=0.2,random_state=50, shuffle=False)
    graphcodebert_X_train, graphcodebert_X_test, graphcodebert_y_train, graphcodebert_y_test = \
        train_test_split(graphcodebert_X, graphcodebert_y, test_size=0.2,random_state=50, shuffle=False)

    del bert_X, bert_y, codebert_X, codebert_y, graphcodebert_X, graphcodebert_y

    #normalize the inputs before doing probing
    bert_norm = Normalization(bert_X_train)
    bert_X_train = bert_norm.norm(bert_X_train)
    bert_X_test = bert_norm.norm(bert_X_test)
    del bert_norm

    codebert_norm = Normalization(codebert_X_train)
    codebert_X_train = codebert_norm.norm(codebert_X_train)
    codebert_X_test = codebert_norm.norm(codebert_X_test)
    del codebert_norm

    graphcodebert_norm = Normalization(graphcodebert_X_train)
    graphcodebert_X_train = graphcodebert_norm.norm(graphcodebert_X_train)
    graphcodebert_X_test = graphcodebert_norm.norm(graphcodebert_X_test)
    del graphcodebert_norm

    #Probeless clustering experiments
    probeless(bert_X_train,bert_y_train,
              codebert_X_train, codebert_y_train,
              graphcodebert_X_train, graphcodebert_y_train)

    #All activations probes
    bert_probe, codebert_probe, graphcodebert_probe, bert_scores, codebert_scores, graphcodebert_scores = all_activations_probe()

    #Layerwise Probes
    layerwise_probes_inference()

    #Important neuron probes
    bert_top_neurons, codebert_top_neurons, graphcodebert_top_neurons = get_imp_neurons()
    get_top_words(bert_top_neurons, codebert_top_neurons, graphcodebert_top_neurons)
    del bert_X_train, bert_X_test, bert_y_train, bert_y_test
    del codebert_X_train, codebert_X_test, codebert_y_train, codebert_y_test
    del graphcodebert_X_train, graphcodebert_X_test, graphcodebert_y_train, graphcodebert_y_test
    #Control task probes
    bert_selectivity, codebert_selectivity, graphcodebert_selectivity = control_task_probes(bert_scores,codebert_scores, graphcodebert_scores)

    return bert_probe, codebert_probe, graphcodebert_probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')

    args = parser.parse_args()
    if args.extract == 'True':
        #bert_activations, codebert_activations = extract_activations()
        bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )
        #bert_activations, codebert_activations = load_extracted_activations()

    #bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    bert_activations, codebert_activations, graphcodebert_activations = load_extracted_activations()

    bert_tokens, codebert_tokens, graphcodebert_tokens =  load_tokens(bert_activations, codebert_activations, graphcodebert_activations)

    bert_probe, codebert_probe, graphcodebert_probe = linear_probes_inference(bert_tokens, bert_activations, codebert_tokens, codebert_activations, graphcodebert_tokens, graphcodebert_activations)
    #neurox.interpretation.utils.print_overall_stats(all_results)

    #Compare linear probe top neurons with probeless neuron ordering


if __name__ == "__main__":
    main()
