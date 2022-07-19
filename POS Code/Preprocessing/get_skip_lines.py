import re
import csv
import os
import argparse
import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader
import neurox.interpretation.utils as utils
import neurox.interpretation.linear_probe as linear_probe
import neurox.interpretation.ablation as ablation
import pickle

bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13) #num_layers is 13 not 768
bert_tokens = data_loader.load_data('codetest.in',
                               'codetest.label',
                               bert_activations,
                               512 # max_sent_length
                              )
