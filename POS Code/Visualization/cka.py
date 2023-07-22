# import torch
# import argparse
# import pickle
# import neurox
# import neurox.data.extraction.transformers_extractor as transformers_extractor
# import neurox.data.loader as data_loader
# import neurox.analysis.corpus as corpus
# import os


# MODEL_NAMES = ['pretrained_BERT',
#                'pretrained_CodeBERT','pretrained_GraphCodeBERT',]
# ACTIVATION_NAMES = {'pretrained_BERT':'bert_activations_train.json',
#                     'pretrained_CodeBERT':'codebert_activations_train.json',
#                     'pretrained_GraphCodeBERT':'graphcodebert_activations_train.json',}

# FOLDER_NAME ="result_all"

# def mkdir_if_needed(dir_name):
#     if not os.path.isdir(dir_name):
#         os.makedirs(dir_name)


# def load_extracted_activations(activation_file_name,activation_folder):
#     #Load activations from json files
#     activations, num_layers = data_loader.load_activations(f"../Experiments/{activation_folder}/{activation_file_name}")
#     return activations


# def HSIC(K, L):
#         """
#         Computes the unbiased estimate of HSIC metric.

#         Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
#         """
#         N = K.shape[0]
#         ones = torch.ones(N, 1).to(self.device)
#         result = torch.trace(K @ L)
#         result += ((ones.t() @ K @ ones @ ones.t() @ L @ ones) / ((N - 1) * (N - 2))).item()
#         result -= ((ones.t() @ K @ L @ ones) * 2 / (N - 2)).item()
#         return (1 / (N * (N - 3)) * result).item()


# def cka(activation1,activation2,model_name1,model_name2):
#     hsic_matrix = torch.zeros(12, 12, 3)
#     for this_sample1 in activation1:
#         # The dimension is seq_len X 9984
#         K = X @ X.t()
#         K.fill_diagonal_(0.0)
#         hsic_matrix[i, :, 0] += self._HSIC(K, K) / num_batches


    


# def main():
#     mkdir_if_needed(f"./{FOLDER_NAME}/")
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--language", default='python')
#     args = parser.parse_args()
#     language = args.language
#     if language == 'python':
#         activation_folder = "activations"
#         src_folder = "src_files"
#     elif language == 'java':
#         activation_folder = "activations_java"
#         src_folder = "src_java"

#     for this_model in MODEL_NAMES:
#         if this_model in ['pretrained_CodeBERT']:
#             print(f"Generate svg files for {this_model}")
#             this_activation_name = ACTIVATION_NAMES[this_model]
#             activations = load_extracted_activations(this_activation_name,activation_folder)
#             print(f"Length of {this_model} activations:",len(activations))
#             _, num_neurons = activations[0].shape
#             for idx in range(len(activations)):
#                 assert activations[idx].shape[1] == num_neurons
#             print(f"The number of neurons for each token in {this_model}:",num_neurons)
#             cka(activations,activations,model_name1=this_model,model_name2=this_model)
#             print("-----------------------------------------------------------------")
#             break

# if __name__ == "__main__":
#     main()


import torch
from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import random

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)
np.random.seed(0)
random.seed(0)

model1 = resnet18(pretrained=True)
model2 = resnet34(pretrained=True)


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

batch_size = 256

dataset = CIFAR10(root='../data/',
                  train=False,
                  download=True,
                  transform=transform)

dataloader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        worker_init_fn=seed_worker,
                        generator=g,)

model1 = resnet50(pretrained=True)
model1.eval()
model2 = wide_resnet50_2(pretrained=True)
model2.eval()

for x1,x2 in zip(dataloader,dataloader):
    print(f"len(x1):{len(x1)}")
    x1_output = model1(x1)
    print(f"x1_output.shape:{x1_output.size()}")
    exit(0)
