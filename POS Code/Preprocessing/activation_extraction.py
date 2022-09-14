import neurox.data.extraction.transformers_extractor as transformers_extractor
import neurox.data.loader as data_loader

transformers_extractor.extract_representations('bert-base-uncased',
    'codetest2_unique.in',
    'bert_activations.json',
    "cuda",
    aggregation="average" #last, first
)

bert_activations, bert_num_layers = data_loader.load_activations('bert_activations.json',13)
bert_tokens = data_loader.load_data('codetest2_unique.in',
                               'codetest2_unique.label',
                               bert_activations,
                               512 # max_sent_length
                              )
