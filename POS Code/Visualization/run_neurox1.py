"""
Created on Tue Apr 12 14:21:21 2022

@author: sharm
"""
import torch
import argparse
import pickle
import neurox
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.analysis.visualization as vis
import neurox.analysis.corpus as corpus
import os


bert_idx = [46,49,75,827]
bert_top_neurons = [2946]
codebert_idx = [1,3,6,17]
codebert_top_neurons = [5585]
graphcodebert_idx = [2,4,8,12,15]
graphcodebert_top_neurons = [9934]


#Extract activations.json files
def extract_activations():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #Extract representations from BERT
    transformers_extractor.extract_representations('bert-base-uncased',
        'codetest2.in',
        'bert_activations.json',
        device=device,
        aggregation="average",#last, first
        decompose_layers=False # we need this to be true to work with different layers
    )

    #Extract representations from CodeBERT
    transformers_extractor.extract_representations('microsoft/codebert-base',
        'codetest2.in',
        'codebert_activations.json',
        device=device,
        aggregation="average", # #last, first
        decompose_layers=False
    )

    #Extract representations from GraphCodeBERT
    transformers_extractor.extract_representations('microsoft/graphcodebert-base',
        'codetest2.in',
        'graphcodebert_activations.json',
        device=device,
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
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/bert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_bert.saveas(name,pretty=True, indent=2)

        for this_neuron in codebert_top_neurons:
            for this_idx in codebert_idx:
                this_svg_codebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     codebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/codebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_codebert.saveas(name,pretty=True, indent=2)

        for this_neuron in graphcodebert_top_neurons:
            for this_idx in graphcodebert_idx:
                this_svg_graphcodebert = vis.visualize_activations(codebert_tokens["source"][this_idx-1],
                                                     graphcodebert_activations[this_idx-1][:, this_neuron],
                                                     filter_fn="top_tokens")
                layer_idx = this_neuron//768
                neuron_idx = this_neuron%768
                name = f"result/graphcodebert_{this_idx-1}_{layer_idx}_{neuron_idx}.svg"
                this_svg_graphcodebert.saveas(name,pretty=True, indent=2)


def get_top_words(bert_tokens,bert_activations,bert_neurons,
                  codebert_tokens=None,codebert_activations=None,codebert_neurons=None,
                  graphcodebert_tokens=None,graphcodebert_activations=None,graphcodebert_neurons=None,
                  dev=True,num_tokens=5):
    if dev:
        for this_neuron in bert_neurons:
            bert_top_words = corpus.get_top_words(bert_tokens, bert_activations,this_neuron,num_tokens)
            print(f"Top words for bert neuron indx {this_neuron}",bert_top_words)
    else:
        for this_neuron in bert_neurons:
            bert_top_words = corpus.get_top_words(bert_tokens, bert_activations,this_neuron,num_tokens)
            print(f"Top words for bert neuron indx {this_neuron}",bert_top_words)
        print("----------------------------------------------------------------")
        for this_neuron in codebert_neurons:
            codebert_top_words = corpus.get_top_words(codebert_tokens, codebert_activations,this_neuron,num_tokens)
            print(f"Top words for codebert neuron indx {this_neuron}",codebert_top_words)
        print("----------------------------------------------------------------")
        for this_neuron in graphcodebert_neurons:
            graphcodebert_top_words = corpus.get_top_words(graphcodebert_tokens, graphcodebert_activations,this_neuron,num_tokens)
            print(f"Top words for graphcodebert neuron indx {this_neuron}",graphcodebert_top_words)


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
    codebert_neurons = [2074,2076,2083,6184,6192,6196,2104,58,69,70,4168,8268,2124,6227,
                        8275,88,8280,6240,4194,6245,2152,8297,6254,8304,2160,8317,8318,4223,
                        8345,4250,6301,6330,187,193,207,6361,219,2271,2273,2275,8429,8431,
                        2300,2312,8467,8468,276,280,8478,4394,2351,4401,8501,6467,2373,2375,
                        330,4428,4438,2400,8545,2406,2410,362,2421,2428,388,8586,8587,395,
                        6546,8598,4505,4512,4520,6583,6596,2515,6619,2525,2528,481,4589,494,
                        2560,2564,6664,6669,535,4635,2590,8735,4641,8738,6691,2599,8748,8766,
                        6719,8769,6726,8775,2633,8778,586,6732,590,6736,6738,595,2647,6750,
                        620,8820,8829,8837,659,667,670,8866,4770,2722,2726,681,2730,6829,
                        2739,696,4811,727,8926,4834,2788,6889,763,6912,4873,8981,6939,8990,
                        806,820,2883,2886,840,9036,2892,4945,2906,7006,862,4960,4961,4964,
                        2916,7023,9072,2928,7028,4982,5002,9137,5051,7105,3015,3018,992,3050,
                        9197,1008,1024,3072,1031,1033,7187,7205,5161,5169,3124,9282,1095,5196,
                        7246,5206,7263,9328,9344,1166,3214,1176,7320,1182,5292,9390,7352,3258,
                        1228,7377,5346,3316,1274,1279,3327,5381,7440,9489,3346,5396,9496,1305,
                        1313,3362,7472,3384,1338,1346,3397,7495,9546,1356,7500,9555,3416,7517,
                        7530,9598,1407,7556,1413,1416,9628,3487,3515,5564,3531,1486,5585,9687,
                        5600,3555,1511,5609,7661,5644,1552,1555,3607,5659,5667,9772,3630,1583,
                        9782,7734,3640,5693,9790,9791,7755,9804,3660,3664,3676,1641,9840,1662,
                        1708,3759,3761,5821,1728,9929,3803,5852,1759,1760,1762,3811,9965,9967,
                        5879,3838,7944,1801,3848,1806,7967,7968,3873,1826,3875,1830,1834,1835,
                        3883,5951,5953,5960,1869,3918,1897,3949,8049,1923,3979,1933,3987,1956,
                        4005,1964,8108,4018,1979,8130,4043,6099,2016,4066,2021,8167,8173,2030,
                        8174,2031,6133]
    graphcodebert_neurons = [6144,8194,6,7,2054,14,15,2068,23,2072,2073,25,8215,30,
                             6195,2100,6210,2124,77,6224,2135,6235,4197,6254,4207,8304,2160,120,
                             2198,6297,4288,8386,4298,8411,219,224,232,8427,236,238,8437,2302,
                             2306,259,2310,2321,4370,4371,280,6433,302,2352,4401,6472,4428,8526,
                             334,6487,4451,8548,361,4458,6512,4464,8564,8565,8575,4497,401,8593,
                             405,2472,424,426,430,433,2486,2495,2497,4548,452,8647,4552,4551,
                             455,459,465,467,4565,6618,479,2529,4583,2540,6645,516,523,2575,
                             2582,6682,2586,6687,4651,2608,579,582,584,6730,8785,606,6755,614,
                             4710,4719,2675,2680,2687,654,4752,8849,6803,660,2712,4761,8860,669,
                             2722,8876,692,2744,2752,707,8914,6885,6890,747,748,6899,8962,4882,
                             8981,2843,2844,2850,2854,2866,9021,2878,835,6981,6987,2892,2893,851,
                             2899,9050,7012,2928,886,2940,9090,2946,912,5018,2972,9117,5025,940,
                             2994,3024,3025,3034,9179,7150,5103,7172,9228,9235,1043,7190,3095,7192,
                             3098,1066,3117,7215,5171,9275,7229,7237,9287,5201,1111,9308,7266,9316,
                             9329,7284,9335,1148,9344,9363,5269,1176,3232,7337,7344,5298,7352,1222,
                             9421,1230,3298,1266,9463,9481,3340,3344,9490,1302,7448,1308,7465,9517,
                             7473,1330,1338,7482,1340,3390,1344,9546,9553,5457,5467,9564,5476,5478,
                             3432,7530,7536,5490,3451,3463,1420,5520,1431,5538,7586,5541,7595,3503,
                             3506,3513,5590,5600,5602,9707,3564,9716,3586,1555,3612,7713,7714,7715,
                             1574,3630,9789,3660,9805,1619,3682,7780,1641,3696,5748,1655,9847,1662,
                             1665,5762,9860,5768,7827,1688,5793,1700,5799,9913,3782,9934,9951,3816,
                             7913,3818,1806,1814,3863,1815,7963,1819,5927,7978,1848,3901,8008,3921,
                             1880,8024,5993,1901,1908,8062,1919,6022,3984,6039,6045,3999,6048,1962,
                             4013,4014,8139,6101,4057,6107,8159,8160,6114,4073,4076,2033,4083]

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
        visualization(bert_tokens, bert_activations, None, None, None, None, True)
        get_top_words(bert_tokens, bert_activations, bert_neurons,
                      None, None, None,
                      None, None, None,
                      True, 5)

    else:
        bert_activations, codebert_activations, graphcodebert_activations = load_extracted_activations(False)
        bert_tokens, codebert_tokens, graphcodebert_tokens =  load_tokens(bert_activations, codebert_activations, graphcodebert_activations,False)
        print("Length of bert_activations:",len(bert_activations))
        print("Length of bert_tokens source:",len(bert_tokens["source"]))
        _, num_neurons = bert_activations[0].shape
        for idx in range(len(bert_activations)):
            assert bert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        print("Length of codebert_activations:",len(codebert_activations))
        print("Length of codebert_tokens source:",len(codebert_tokens["source"]))
        _, num_neurons = codebert_activations[0].shape
        for idx in range(len(codebert_activations)):
            assert codebert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        print("Length of graphcodebert_activations:",len(graphcodebert_activations))
        print("Length of graphcodebert_tokens source:",len(graphcodebert_tokens["source"]))
        _, num_neurons = graphcodebert_activations[0].shape
        for idx in range(len(graphcodebert_activations)):
            assert graphcodebert_activations[idx].shape[1] == num_neurons
        print("The number of neurons for each token:",num_neurons)

        visualization(bert_tokens, bert_activations,
                      codebert_tokens,codebert_activations,
                      graphcodebert_tokens,graphcodebert_activations,
                      False)
        get_top_words(bert_tokens, bert_activations, bert_neurons,
                      codebert_tokens, codebert_activations, codebert_neurons,
                      graphcodebert_tokens, graphcodebert_activations, graphcodebert_neurons,
                      False, 5)


if __name__ == "__main__":
    main()
