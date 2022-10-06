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


MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',
               'finetuned_defdet_CodeBERT','finetuned_defdet_GraphCodeBERT',
               'finetuned_clonedet_CodeBERT','finetuned_clonedet_GraphCodeBERT']
ACTIVATION_NAMES = ['bert_activations.json',
                    'codebert_activations.json','graphcodebert_activations.json',
                    'codebert_defdet_activations.json','graphcodebert_defdet_activations.json',
                    'codebert_clonedet_activations1.json','graphcodebert_clonedet_activations1.json']

class Normalization:
    def __init__(self,df):
        self.var_mean = np.mean(df,axis=0)
        self.var_std = np.std(df,axis=0)

    def norm(self,df):
        norm_df = (df-self.var_mean)/self.var_std
        return norm_df


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


def load_extracted_activations(activation_file_name):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(activation_file_name,13)
    return activations

def load_tokens(activations):
    #Load tokens and sanity checks for parallelism between tokens, labels and activations
    tokens = data_loader.load_data('codetest2_unique.in',
                                   'codetest2_unique.label',
                                   activations,
                                   512 # max_sent_length
                                  )
    return tokens


def linear_probes_inference(tokens, activations,model_name):
    ''' Returns models and accuracy(score) of the probes trained on entire activation space '''

    def get_mappings(tokens,activations):
        ''' Get mappings for all models'''
        X, y, mapping = utils.create_tensors(tokens, activations, 'NAME') #mapping contains tuple of 4 dictionaries
        label2idx, idx2label, src2idx, idx2src = mapping

        return X, y, label2idx, idx2label, src2idx, idx2src

    def all_activations_probe(X_train,y_train,X_test,y_test,idx2label,model_name):
        #Train the linear probes (logistic regression) - POS(code) tagging
        probe = linear_probe.train_logistic_regression_probe(X_train, y_train, lambda_l1=0.001, lambda_l2=0.001)

        #Get scores of probes
        print(f"Accuracy on the test set of probing {model_name} of all layers:")
        scores = linear_probe.evaluate_probe(probe, X_test, y_test, idx_to_class=idx2label)
        print(scores)
        X_test_baseline = np.zeros_like(X_test)
        print(f"Accuracy on the test set of {model_name} model using the intercept:")
        linear_probe.evaluate_probe(probe, X_test_baseline, y_test, idx_to_class=idx2label)
        return probe, scores

    def get_imp_neurons(X_train,y_train,X_test,y_test,probe,label2idx,idx2label,model_name):
        ''' Returns top 2% neurons for each model'''

        #Top neurons
        top_neurons, top_neurons_per_class = linear_probe.get_top_neurons(probe, 0.02, label2idx)
        print(f"{model_name} top neurons")
        print(repr(top_neurons))
        print(f"{model_name} top neurons per class")
        print(top_neurons_per_class)

        #Train probes on top neurons and save them
        X_selected = ablation.filter_activations_keep_neurons(X_train, top_neurons)
        print("The shape of selected features",X_selected.shape)
        probe_selected = linear_probe.train_logistic_regression_probe(X_selected, y_train, lambda_l1=0.001, lambda_l2=0.001)
        del X_selected
        X_selected_test = ablation.filter_activations_keep_neurons(X_test, top_neurons)
        print(f"Accuracy on the test set of {model_name} model on top 2% neurons:")
        linear_probe.evaluate_probe(probe_selected, X_selected_test, y_test, idx_to_class=idx2label)
        del X_selected_test

        ordering, cutoffs = linear_probe.get_neuron_ordering(probe, label2idx)
        X_selected = ablation.filter_activations_keep_neurons(X_train, ordering[:200])
        probe_selected = linear_probe.train_logistic_regression_probe(X_selected, y_train, lambda_l1=0.001, lambda_l2=0.001)
        del X_selected
        X_selected_test = ablation.filter_activations_keep_neurons(X_test, ordering[:200])
        print(f"Accuracy on the test set of {model_name} model on top 200 neurons:")
        return top_neurons

    def get_top_words(top_neurons,tokens,activations,model_name):
        #relate neurons to corpus elements like words and sentences
        """
        pretrained_Bert:
        idx: 1169, NAME
        [('nh', 1.0), ('ts', 0.9368543327236498), ('bk', 0.7797181845932368), ('leaders', 0.7338799023084398), ('bias', 0.7298205393199121)]
        idx: 4100 STAR
        [('Orientation', 1.0), ('*', 0.9590381176401255), ('selector', 0.8984144128018782), ('Mock', 0.8927492917214652), ('easter', 0.8893109745730953)]
        idx: 484 PERCENT
        [('%', 1.0), ('plot', 0.9291268513168732), ('{', 0.9155056691047887), ('uniform', 0.8875826274073034), ('st', 0.8838933131118305)]

        pretrained_CodeBert:
        idx: 83, RSQB
        [(']', 1.0), ('gen', 0.8978395588370207), ('genes', 0.8583926365691852), ('src', 0.85231653219694), ('release', 0.8498507896190105)]
        idx: 284 NUMBER
        [('319', 1.0), ('102', 0.982483609803209), ('625', 0.9824103180473372), ('192', 0.9813310622249785), ('1024', 0.9724426546786248)]
        idx: 2002, STRING
        [('emitter', 1.0), ('easter', 0.9536935621647816), ('"instance"', 0.8215065421481901), ('"driven"', 0.8134414348372471), ('"shader"', 0.7777029467232262)]
        idx: 2454, LBRACE
        [('{', 1.0), ('WAYS', 0.9333973654403263), ('API', 0.8976658221825284), ('PORT', 0.7935343612534739), ('area_rad', 0.7566464641066661)]
        idx: 8991, MINUS
        [('-', 1.0), ('@', 0.783963625194319), ('filter_by', 0.7684178121629761), ('"headtextcolor"', 0.7650026602484139), ('"textcolor"', 0.7589861261515543)]

        pretrained_GraphCodeBert:
        idx: 83, RSQB
        [(']', 1.0), ('lookup', 0.9501715866645286), ('communities', 0.9482650915781232), ('gen', 0.9356474625343222), ('WAYS', 0.8742862253662748)]
        idx: 284 NUMBER
        [('120', 1.0), ('503', 0.9669304378066406), ('302', 0.9588958846028263), ('192', 0.9333921868105192), ('8000', 0.9087051688114941)]
        idx: 8962 MINUS
        [('-', 1.0), ('DataFrame', 0.9650571679043924), ('pandas', 0.9459861451873764), ('authorityCertificate', 0.9149520446943905), ('psycopg2', 0.8922085426479229)]
        idx: 5586 NOTEQUAL
        [('!=', 1.0), ('external_audience', 0.7733831302458536), ('ix', 0.7636406579259619), ('Output', 0.7596070007130915), ('none_on_404', 0.7425271132925706)]
        idx: 2059, AT
        [('@', 1.0), ('Delay', 0.9338737596823018), ('Bytes', 0.8771050612134269), ('GENERATOR', 0.8247529369592388), ('ACLU_NJ', 0.8237272451291417)]

        finetuned_defdet_CodeBert:
        idx: 83, RSQB
        [(']', 1.0), ('gen', 0.8933596401126435), ('genes', 0.8567785473115318), ('release', 0.84948227849402), ('src', 0.8472325904542191)]
        idx: 284 NUMBER
        [('319', 1.0), ('192', 0.987243380345884), ('102', 0.9864111437608304), ('625', 0.9828795237703138), ('302', 0.9758179194986557)]
        idx: 124, STRING
        [('friend', 1.0), ('heads', 0.9486152914141273), ('buddy', 0.924081501250217), ('head', 0.8602763118892631), ('neighbor', 0.8341647234318657)]
        idx:2454, LBRACE
        [('WAYS', 1.0), ('{', 0.9676186034748889), ('API', 0.8682861343724128), ('NODE', 0.8533325859481167), ('WAY', 0.8126914158589422)]
        idx: 1448, MINUS
        [('public', 1.0), ('rest', 0.9109589355441492), ('masked', 0.9006628025649864), ('Standard', 0.8962497184995595), ('sam', 0.8935695976175985)]

        finetuned_defdet_GraphCodeBert
        idx: 83, RSQB
        [(']', 1.0), ('lookup', 0.9520670112357633), ('communities', 0.9516259594529405), ('gen', 0.9417809282136829), ('WAYS', 0.8738937478389772)]
        idx: 284,NUMBER
        [('120', 1.0), ('503', 0.9704695747902121), ('302', 0.9629492590019171), ('192', 0.9392358588028403), ('8000', 0.9076185419150461)]
        idx: 1165, MINUS
        [('192', 1.0), ('36', 0.9162604016157989), ('cont', 0.9131234991889443), ('checkout', 0.9051554674962018), ('par', 0.9006388788613352)]
        idx: 5586, NOTEQUAL
        [('!=', 1.0), ('none_on_404', 0.8464201147869352), ('external_audience', 0.8272049098253852), ('784', 0.8269285829242635), ('"http://www.w3.org/2002/07/owl#"', 0.8072960744727502)]
        idx: 2059, AT
        [('@', 1.0), ('Delay', 0.8671508822683335), ('0.', 0.8161755516796887), ('Bytes', 0.8077888178178824), ('ACLU_NJ', 0.8028997123186968)]

        finetuned_clonedet_CodeBert
        idx:83, RSQB
        [(']', 1.0), ('gen', 0.9103672404100732), ('genes', 0.8666678476610976), ('src', 0.8639488002507691), ('release', 0.8500897270088731)]
        idx:284, NUMBER
        [('319', 1.0), ('625', 0.9801414885079729), ('102', 0.9755307652871615), ('192', 0.9739028855615162), ('302', 0.965847503748241)]
        idx: 507, STRING
        [('indexwidth', 1.0), ('demo', 0.8920956314387782), ('MAXLAT', 0.8762036152757903), ('extra', 0.8739816031297483), ('ComboBox', 0.8678652412301193)]
        idx: 652, LBRACE
        [('intern', 1.0), ('{', 0.9469888681803765), ('principal', 0.9182474371667484), ('Off', 0.9007017040796614), ('factories', 0.8785951069775485)]
        idx: 1176, MINUS
        [('Server', 1.0), ('Authorization', 0.9775819074987048), ('serializer', 0.9113232315259349), ('Provider', 0.9004882831511033), ('servers', 0.8879392656770141)]

        finetuned_clonedet_GraphCodeBert
        idx: 83, RSQB
        [(']', 1.0), ('lookup', 0.9644543905945676), ('gen', 0.9617436575077136), ('communities', 0.9544403309360746), ('attribute', 0.8727061690669303)]
        idx: 284, NUMBER
        [('120', 1.0), ('503', 0.9668672585840505), ('302', 0.9543204867671596), ('192', 0.937074441146183), ('8000', 0.915807192353721)]
        idx: 1165, MINUS
        [('192', 1.0), ('cont', 0.8778030380130084), ('checkout', 0.8752734090324291), ('36', 0.8584594706337377), ('authentication', 0.8474504971288491)]
        idx: 224, NOTEQUAL
        [('Uri', 1.0), ('framing', 0.9439110630472592), ('111', 0.9127593170247624), ('recipe', 0.906373307643622), ('enclosure', 0.8919352986184027)]
        idx: 2059, AT
        [('straight', 1.0), ('ACLU_NJ', 0.942600446611766), ('@', 0.860884857370497), ('POST', 0.8565302777694044), ('through', 0.8317164966072599)]
        """
        print(f"{model_name} top words")
        for neuron in top_neurons:
            top_words = corpus.get_top_words(tokens, activations, neuron, num_tokens=5)
            print(f"Top words for {model_name} neuron indx {neuron}",top_words)

    def layerwise_probes_inference(X_train,y_train,X_test,y_test,idx2label,model_name):
        ''' Returns models and accuracy(score) of the probes trained on activations from different layers '''
        for i in range(13):
            print(f"{model_name} Layer", i)
            layer_train = ablation.filter_activations_by_layers(X_train, [i], 13)
            layer_probe = linear_probe.train_logistic_regression_probe(layer_train, y_train, lambda_l1=0.001, lambda_l2=0.001)
            del layer_train
            pickle.dump(layer_probe, open(f"{model_name}_layer{i}_probe.sav", 'wb'))
            layer_test = ablation.filter_activations_by_layers(X_test, [i], 13)
            linear_probe.evaluate_probe(layer_probe, layer_test, y_test, idx_to_class=idx2label)
            del layer_test
            del layer_probe

    def control_task_probes(tokens,activations,original_scores,model_name):
        print(f"Creating control dataset for {model_name} POS tagging task")
        # [ct_tokens] = ct.create_sequence_labeling_dataset(tokens, sample_from='uniform')
        [ct_tokens] = ct.create_sequence_labeling_dataset(tokens, sample_from='same')
        X_ct, y_ct, mapping_ct = utils.create_tensors(ct_tokens, activations, 'NAME')
        label2idx_ct, idx2label_ct, src2idx_ct, idx2src_ct = mapping_ct

        X_ct,y_ct,label2idx_ct,idx2label_ct=filter_by_frequency(X_ct,y_ct,label2idx_ct,idx2label_ct,40,model_name+'_control_task')

        X_ct_train, X_ct_test, y_ct_train, y_ct_test = \
            train_test_split(X_ct, y_ct, test_size=0.2,random_state=50, shuffle=True)
        # normalization
        ct_norm = Normalization(X_ct_train)
        X_ct_train = ct_norm.norm(X_ct_train)
        X_ct_test = ct_norm.norm(X_ct_test)
        del ct_norm

        ct_probe = linear_probe.train_logistic_regression_probe(X_ct_train, y_ct_train, lambda_l1=0.001, lambda_l2=0.001)
        print(f"Accuracy on the test set of {model_name} control model:")
        ct_scores = linear_probe.evaluate_probe(ct_probe, X_ct_test, y_ct_test, idx_to_class=idx2label_ct)
        selectivity = original_scores['__OVERALL__'] - ct_scores['__OVERALL__']
        print(f'{model_name} Selectivity (Diff. between true task and probing task performance): ', selectivity)
        del ct_scores
        X_ct_test_baseline = np.zeros_like(X_ct_test)
        print(f"Accuracy on the test set of {model_name} control model using the intercept:")
        linear_probe.evaluate_probe(ct_probe, X_ct_test_baseline, y_ct_test, idx_to_class=idx2label_ct)

        top_neurons_ct, _ = linear_probe.get_top_neurons(ct_probe, 0.02, label2idx_ct)
        X_selected_ct = ablation.filter_activations_keep_neurons(X_ct_train, top_neurons_ct)
        probe_selected_ct = linear_probe.train_logistic_regression_probe(X_selected_ct, y_ct_train, lambda_l1=0.001, lambda_l2=0.001)
        del X_selected_ct
        X_selected_test_ct = ablation.filter_activations_keep_neurons(X_ct_test, top_neurons_ct)
        print(f"Accuracy on the test set of {model_name} control model on top neurons:")
        linear_probe.evaluate_probe(probe_selected_ct, X_selected_test_ct, y_ct_test, idx_to_class=idx2label_ct)
        del X_selected_test_ct

        ordering, cutoffs = linear_probe.get_neuron_ordering(ct_probe, label2idx_ct)
        X_selected_ct = ablation.filter_activations_keep_neurons(X_ct_train, ordering[:200])
        probe_selected_ct = linear_probe.train_logistic_regression_probe(X_selected_ct, y_ct_train, lambda_l1=0.001, lambda_l2=0.001)
        del X_selected_ct
        X_selected_test_ct = ablation.filter_activations_keep_neurons(X_ct_test, ordering[:200])
        print(f"Accuracy on the test set of {model_name} control model on top 200 neurons:")
        linear_probe.evaluate_probe(probe_selected_ct, X_selected_test_ct, y_ct_test, idx_to_class=idx2label_ct)
        del X_selected_test_ct
        del X_ct_train, y_ct_train, X_ct_test, y_ct_test
        del ct_probe

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
        import collections
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
        distribution = {k: v/len(y) for k, v in sorted(count.items(), key=lambda item: item[1],reverse=True)}
        print(f"{model_name} distribution after trauncating:")
        print(distribution)
        return X,y,label2idx,idx2label


    #Get mappings
    X, y, label2idx, idx2label, src2idx, idx2src = get_mappings(tokens,activations)

    X, y, label2idx, idx2label = filter_by_frequency(X,y,label2idx,idx2label,40,model_name)

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2,random_state=50, shuffle=True)

    del X, y

    #normalize the inputs before doing probing
    norm = Normalization(X_train)
    X_train = norm.norm(X_train)
    X_test = norm.norm(X_test)
    del norm

    #Probeless clustering experiments
    # probeless(X_train,y_train,model_name)

    #All activations probes
    probe, scores = all_activations_probe(X_train,y_train,X_test, y_test,idx2label,model_name)

    #Layerwise Probes
    # layerwise_probes_inference(X_train,y_train,X_test,y_test,idx2label,model_name)

    #Important neuron probes
    top_neurons = get_imp_neurons(X_train,y_train,X_test,y_test,probe,label2idx,idx2label,model_name)
    # get_top_words(top_neurons,tokens,activations,model_name)
    del X_train, X_test, y_train, y_test
    #Control task probes
    selectivity = control_task_probes(tokens,activations,scores,model_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')

    args = parser.parse_args()
    if args.extract == 'True':
        bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )

    for this_model, this_activation_name in zip(MODEL_NAMES,ACTIVATION_NAMES):
        print(f"Anayzing {this_model}")
        activations = load_extracted_activations(this_activation_name)

        tokens =  load_tokens(activations)

        linear_probes_inference(tokens,activations,this_model)
        print("----------------------------------------------------------------")
        break


if __name__ == "__main__":
    main()
