"""
Created on Tue Apr 12 14:21:21 2022

@author: sharm
"""
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.analysis.visualization as vis
import neurox.analysis.corpus as corpus
import os

layer = 0
bert_idx = [46,49,75,827]
bert_top_neurons = [2946]
codebert_idx = [1,3,6,17]
codebert_top_neurons = [5585]
graphcodebert_idx = [2,4,8,12,15]
graphcodebert_top_neurons = [9934]


#Extract activations.json files
def extract_activations():
    #Extract representations from BERT
    transformers_extractor.extract_representations('bert-base-uncased',
        'codetest2.in',
        'bert_activations.json',
        aggregation="average",#last, first
        decompose_layers=False # we need this to be true to work with different layers
    )

    #Extract representations from CodeBERT
    transformers_extractor.extract_representations('microsoft/codebert-base',
        'codetest2.in',
        'codebert_activations.json',
        aggregation="average", # #last, first
        decompose_layers=False
    )

    #Extract representations from GraphCodeBERT
    transformers_extractor.extract_representations('microsoft/graphcodebert-base',
        'codetest2.in',
        'graphcodebert_activations.json',
        aggregation="average",#last, first
        decompose_layers=False
    )

    return(load_extracted_activations())


def load_extracted_activations(dev):
    if dev:
        bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
        return bert_activations
    else:
        #Load activations from json files
        bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
        codebert_activations, codebert_num_layers = data_loader.load_activations('codebert_activations.json',13) #num_layers is 13 not 768
        graphcodebert_activations, graphcodebert_num_layers = data_loader.load_activations('graphcodebert_activations.json',13)

        return bert_activations, codebert_activations, graphcodebert_activations


def load_tokens(bert_activations,codebert_activations=None, graphcodebert_activations=None,dev=True):
    if dev:
        bert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )
        return bert_tokens
    else:
        #Load tokens and sanity checks for parallelism between tokens, labels and activations
        bert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       bert_activations,
                                       512 # max_sent_length
                                      )

        codebert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       codebert_activations,
                                       512 # max_sent_length
                                      )

        graphcodebert_tokens = data_loader.load_data('codetest2.in',
                                       'codetest2.label',
                                       graphcodebert_activations,
                                       512 # max_sent_length
                                      )


        return bert_tokens, codebert_tokens, graphcodebert_tokens



def visualization(bert_tokens, bert_activations,
                  codebert_tokens = None, codebert_activations = None,
                  graphcodebert_tokens = None, graphcodebert_activations = None,
                  dev=True):
    # viz_bert = TransformersVisualizer('bert-base-uncased')
    # viz_codebert = TransformersVisualizer('microsoft/codebert-base')
    # viz_graphcoderbert = TransformersVisualizer('microsoft/graphcodebert-base')

    if dev:
        # starting from 1.
        for this_neuron in bert_top_neurons:
            for this_idx in bert_idx:
                this_svg_bert = vis.visualize_activations(bert_tokens["source"][this_idx-1],
                                                     bert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                name = f"result/bert_{this_idx-1}_{layer}_{this_neuron}.svg"
                this_svg_bert.saveas(name,pretty=True, indent=2)
    else:
        # starting from 1.
        for this_neuron in bert_top_neurons:
            for this_idx in bert_idx:
                this_svg_bert = vis.visualize_activations(bert_tokens["source"][this_idx-1],
                                                     bert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                name = f"result/bert_{this_idx-1}_{layer}_{this_neuron}.svg"
                this_svg_bert.saveas(name,pretty=True, indent=2)

        for this_neuron in codebert_top_neurons:
            for this_idx in codebert_idx:
                this_svg_codebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     codebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                name = f"result/codebert_{this_idx-1}_{layer}_{this_neuron}.svg"
                this_svg_codebert.saveas(name,pretty=True, indent=2)

        for this_neuron in graphcodebert_top_neurons:
            for this_idx in graphcodebert_idx:
                this_svg_graphcodebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     graphcodebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                name = f"result/graphcodebert_{this_idx-1}_{layer}_{this_neuron}.svg"
                this_svg_graphcodebert.saveas(name,pretty=True, indent=2)


def get_top_words(bert_tokens,bert_activations,bert_neurons,
                  codebert_tokens=None,codebert_activations=None,codebert_neuron=None,
                  graphcodebert_tokens=None,graphcodebert_activations=None,graphcodebert_neuron=None,
                  dev=True,num_tokens=5):
    if dev:
        for this_neuron in bert_neurons:
            bert_top_words = corpus.get_top_words(bert_tokens, bert_activations,this_neuron,num_tokens)
            print(f"Top words for bert neuron indx {this_neuron}",bert_top_words)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract",choices=('True','False'), default='False')
    parser.add_argument("--dev",choices=('True','False'), default='True')
    args = parser.parse_args()
    if args.extract == 'True':
        bert_activations, codebert_activations,graphcodebert_activations = extract_activations()
    else:
        print("Getting activations from json files. If you need to extract them, run with --extract=True \n" )
    # bert
    # idx 7269:
    # [('20', 1.0), ('200', 0.8789008386983315), ('5000', 0.8719975643707852), ('100000', 0.7574842427994176), ('10000', 0.7223461119136357)]
    # idx 1962
    # [('Cube', 1.0), ('cube', 0.9007572097399958), ('children', 0.8544416047592817), ('size', 0.850839966186369), ('child', 0.8098641075300375)]
    # idx 4032
    # [('190', 1.0), ('150', 0.9441334717348694), ('17', 0.8987034576046211), ('19', 0.8508893737865472), ('401', 0.838905788473958)]
    bert_neurons = [6173,4131,37,6181,41,2091,2095,6205,2112,64,4171,75,8274,8275,
                   85,6231,4195,8297,8306,6259,6261,129,134,138,4244,152,2205,173,
                   183,6329,4284,195,2246,6343,4302,4312,6364,8421,8437,6389,6392,249,
                   6393,2301,260,2317,6416,278,292,4393,305,308,6453,4405,2362,8515,
                   6468,324,2375,2406,4457,8553,2411,2428,8578,6534,6540,408,8611,2470,
                   436,6588,6596,4557,2514,4568,2520,2523,495,8690,6644,2551,8696,2558,
                   6663,6664,540,8732,545,4645,8756,573,8765,2624,6721,2627,580,6735,
                   601,602,4697,4706,623,2685,2689,8835,643,2693,8851,2718,2719,2723,
                   4779,684,687,4795,8895,6855,6859,4813,2766,718,8920,4830,6879,6880,
                   2789,744,745,2800,6897,760,763,4860,4865,2833,6958,6979,9052,7007,
                   4990,2945,2946,9093,7050,9101,7062,5023,7082,951,955,959,978,7128,
                   5080,7157,3065,5119,1023,7178,9235,9248,7202,3108,1071,3126,3141,1099,
                   5203,3161,7260,7269,9329,1141,3194,3209,7310,1170,3238,3239,7347,3255,
                   9400,7351,3264,7375,3282,9428,7384,5341,7394,7403,9463,1271,7417,1273,
                   3343,9494,1304,3357,7459,5413,5423,9519,9532,5451,1355,3417,9564,7528,
                   1391,9585,5503,7554,7558,5511,1417,7568,9622,9625,9627,9633,7588,5546,
                   5563,5565,7614,5572,5574,3527,7624,3534,1500,7645,5600,7649,7656,3561,
                   9706,3570,5625,1537,3588,1562,1566,7717,3628,9772,9777,3637,3649,5700,
                   7750,1608,7754,1618,1619,5715,7772,5730,9828,9829,7785,5746,9863,9866,
                   1675,1684,7843,5803,3759,9908,1721,9913,3773,5828,9926,1735,3784,5840,
                   3796,5848,9968,7928,3832,9980,1796,3855,3862,3871,5923,3877,7987,7988,
                   5942,1876,5980,5984,1890,5991,8043,3949,3955,6012,3996,1950,8101,1962,
                   6059,1977,1980,4032,4034,6090,4055,8152,6106,6115,4078,2035,4084,6133,
                   2045]
 # MINEQUAL
    if args.dev == 'True':
        bert_activations = load_extracted_activations(True)
        bert_tokens =  load_tokens(bert_activations, None, None,True)
        print("Length of bert_activations:",len(bert_activations))
        print("Length of bert_tokens source:",len(bert_tokens["source"]))
        _, num_neurons = bert_activations[0].shape
        for idx in range(len(bert_activations)):
            assert bert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)
        print("Done!")
        exit(0)

        visualization(bert_tokens, bert_activations, None, None, None, None, True)
        get_top_words(bert_tokens, bert_activations, bert_neurons,
                      None, None, None,
                      None, None, None,
                      True, 5)

    else:
        bert_activations, codebert_activations, graphcodebert_activations = load_extracted_activations(False)
        bert_tokens, codebert_tokens, graphcodebert_tokens =  load_tokens(bert_activations, codebert_activations, graphcodebert_activations,False)
        visualization(bert_tokens, bert_activations,
                      codebert_tokens,codebert_activations,
                      graphcodebert_tokens,graphcodebert_activations,
                      False)


if __name__ == "__main__":
    main()
