# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import collections
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model #Which model is this?
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                          AutoConfig, AutoModel, AutoTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'auto': (AutoConfig, AutoModel, AutoTokenizer)
}



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 idx,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx=str(idx)
        self.label=label

        
def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['func'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['idx'],js['target'])

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js=json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js,tokenizer,args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                    logger.info("*** Example ***")
                    logger.info("idx: {}".format(idx))
                    logger.info("label: {}".format(example.label))
                    logger.info("input_tokens: {}".format([x.replace('\u0120','_') for x in example.input_tokens]))
                    logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):       
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)
            

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract(args, dataset, model, tokenizer, prefix="", evaluate=True, debug=True):
    """ Extract activations from the model """ 
    args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    sampler = RandomSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    
    dataloader = DataLoader(dataset, sampler=sampler, 
                                  batch_size=args.batch_size,num_workers=4,pin_memory=True)

    #dataset = load_and_cache_examples(args, args.task_name, tokenizer, ttype='train')

    # args.batch_size = args.per_gpu_batch_size * max(1, args.n_gpu)
    # # Note that DistributedSampler samples randomly
    # sampler = SequentialSampler(dataset) if args.local_rank == -1 else DistributedSampler(dataset)
    # dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size)

    print(dataloader)

    layer_indexes = [int(x) for x in args.layers.split(",")]

    logger.info("***** Running extraction {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Extracting layers = %s", ",".join(map(str, layer_indexes)))
    model.eval()
    with open(args.output_file, "w", encoding='utf-8') as writer:
        unique_id = 0
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to(args.device) for t in batch)
            print("printing batch")	       
            with torch.no_grad():
                inputs = {'input_ids':      batch[0]}
                           # 'labels':       batch[1]}
                labels = batch[1]                
                if args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
                outputs = model(**inputs,output_hidden_states=True)
                print(outputs)
                all_inputs = batch[0] 
                layer_outputs = outputs[2]  #[0] last hidden state  [1] pooler_output logits [2] hidden layer outputs incl initial embedding outputs
                layer_outputs = [x.detach().cpu().numpy() for x in layer_outputs]
                print("layer outputs", layer_outputs)
		
                for example_idx in range(len(all_inputs)):
                    if debug: print("Processing example no %d" % (unique_id))
                    start_time = time.time()
                    output_json = collections.OrderedDict()

                    # Iterate over each example in the batch
                    input_ids = all_inputs[example_idx].tolist()
                    output_json["linex_index"] = unique_id
                    all_out_features = []

                    tokens = list(enumerate(tokenizer.convert_ids_to_tokens(input_ids)))
                    assert len(tokens) == len(input_ids)
                    if debug:
                        print("All tokens")
                        print(tokens)
                    if args.model_type == 'bert':
                        tokens = [(i,t) for i,t in tokens if t != '[PAD]']
                    elif args.model_type == 'xlnet':
                        tokens = [(i,t) for i,t in tokens if t != '<pad>']
                    elif args.model_type == 'auto':
                        tokens = [(i,t) for i,t in tokens if t != '<pad>']
                    elif args.model_type == 'distilbert':
                        tokens = [(i,t) for i,t in tokens if t != '[PAD]']
                     
                    #Only get CLS tokens or first token
                    if args.sentence_only and args.model_type == 'bert':
                        tokens = [(i,t) for i,t in tokens if t == '[CLS]']
                    if args.sentence_only and args.model_type == 'roberta':
                        tokens = [(i,t) for i,t in tokens if t == '<s>']
                    if args.sentence_only and args.model_type == 'auto':
                        tokens = [(i,t) for i,t in tokens if t == '<s>']
                    if args.sentence_only and args.model_type == 'xlnet':
                        tokens = [(i,t) for i,t in tokens if t == '<cls>']
                    if args.sentence_only and args.model_type == 'distilbert':
                        tokens = [(i,t) for i,t in tokens if t == '[CLS]']

                    if debug:
                        print("Extracting tokens:")
                        print(tokens)

                    for token_idx, token in tokens:
                        all_layers = []
                        for j, layer_idx in enumerate(layer_indexes):
                            layers = collections.OrderedDict()
                            print(layer_idx, example_idx, token_idx)
                            layers["index"] = layer_idx
                            layers["values"] = [ round(x.item(), 6) for x in layer_outputs[layer_idx][example_idx][token_idx]] #[layer_idx][example_idx][token_idx]
                            all_layers.append(layers)
                        print("all_layers", all_layers)
                        out_features = collections.OrderedDict()
                        out_features["token"] = token
                        out_features["layers"] = all_layers
                        all_out_features.append(out_features)
                    output_json["features"] = all_out_features
                    end_time = time.time()
                    if debug: print("Computed in %d s" %(end_time-start_time))
                    writer.write(json.dumps(output_json) + "\n")
                    end_time = time.time()
                    if debug: print("Saved in %d s" %(end_time-start_time))

                    unique_id += 1
                        
                        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
#    parser.add_argument("--output_dir", default=None, type=str, required=True,
 #                       help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--output_file", default=None, type=str, required=True,
                        help="The output file where features will be saved.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")

    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")

    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
                        
    #Training parameters
    parser.add_argument("--batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")

    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # parser.add_argument('--epoch', type=int, default=42,
    #                     help="random seed for initialization")
                    
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--sentence_only', action='store_true', help="For Extracting only [CLS] token embeddings")
    parser.add_argument('--layers', type=str, default='', help="Comma separated list of layers to extract. 0: Embeddings, 1-12: Normal layers")


    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    args.per_gpu_batch_size=args.batch_size//args.n_gpu
    #args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    #deleted

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,
					    ignore_mismatched_sizes=True)    
    else:
        model = model_class(config)

    #Pretrained model with finetuned layer added
#    model=Model(model,config,tokenizer,args)
#    if args.local_rank == 0:
#        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Extraction parameters %s", args)

    model.to(args.device)
    # Training -- dataset created here --make extraction changes here
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args,args.data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

    
        #train(args, train_dataset, model, tokenizer)
        extract(args, train_dataset, model, tokenizer, evaluate=False) #Add dataset too, no load and cache
    else:
        extract(args, train_dataset, model, tokenizer, evaluate=True)


if __name__ == "__main__":
    main()


