# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss

class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, input_ids=None,labels=None): 
        logits=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1),output_hidden_states=True)
        print("outputs = ",outputs.logits, outputs.hidden_states)
        if self.args.model_type=="gpt2":
            hidden_states=outputs[2]
        else:
            hidden_states=outputs[1]#1 for BERT and 2 for gpt as 1 is past_values
        
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob,hidden_states #labels=None for test or extract
            
      
        
 
