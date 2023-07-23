import argparse
import neurox.data.loader as data_loader
import os
import numpy as np
import matplotlib.pyplot as plt

MODEL_NAMES = ['pretrained_BERT',
               'pretrained_CodeBERT','pretrained_GraphCodeBERT',]
ACTIVATION_NAMES = {'pretrained_BERT':'bert_activations_train.json',
                    'pretrained_CodeBERT':'codebert_activations_train.json',
                    'pretrained_GraphCodeBERT':'graphcodebert_activations_train.json',}

FOLDER_NAME ="result_all"

N_LAYERs = 13
N_NEUROSN_PER_LAYER = 768
N_SAMPLES = 5000

def mkdir_if_needed(dir_name):
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)


def load_extracted_activations(activation_file_name,activation_folder):
    #Load activations from json files
    activations, num_layers = data_loader.load_activations(f"../Experiments/{activation_folder}/{activation_file_name}")
    return activations


def HSIC(K, L):
        """
        Computes the unbiased estimate of HSIC metric.

        Reference: https://arxiv.org/pdf/2010.15327.pdf Eq (3)
        """
        N = K.shape[0]
        ones = np.ones((N, 1))
        result = np.trace(K @ L)
        result += (ones.transpose() @ K @ ones @ ones.transpose() @ L @ ones) / ((N - 1) * (N - 2))
        result -= (ones.transpose() @ K @ L @ ones) * 2 / (N - 2)
        return 1 / (N * (N - 3)) * result[0,0]


def plot_results(hsic_matrix,save_path,title):
        fig, ax = plt.subplots()
        im = ax.imshow(hsic_matrix, origin='lower', cmap='magma')
        ax.set_xlabel(f"Layers {self.model2_info['Name']}", fontsize=15)
        ax.set_ylabel(f"Layers {self.model1_info['Name']}", fontsize=15)

        ax.set_title(f"{title}", fontsize=18)

        add_colorbar(im)
        plt.tight_layout()

        plt.savefig(save_path, dpi=300)

        plt.show()


def cka(activation1,n_samples):
    hsic_matrix = np.zeros((N_LAYERs, N_LAYERs, 3))
    X = np.array([this_token for this_sample in activation1 for this_token in this_sample])
    del activation1
    np.random.seed(2)
    random_choice = np.random.choice(len(X),size=n_samples,replace=False)
    random_choice = sorted(random_choice)
    X = X[random_choice]
    num_batches = 1
    
    
    for i in range(N_LAYERs):
        index = i*N_NEUROSN_PER_LAYER
        this_X = X[:,index:index+N_NEUROSN_PER_LAYER]
        # The dimension is seq_len X 9984
        K = this_X @ this_X.transpose()
        np.fill_diagonal(K,0.0)
        hsic_matrix[i, :, 0] += HSIC(K, K) / num_batches

        for j in range(N_LAYERs):
            index = j*N_NEUROSN_PER_LAYER
            this_Y = X[:,index:index+N_NEUROSN_PER_LAYER]
            L = this_Y @ this_Y.transpose()
            np.fill_diagonal(L,0)

            hsic_matrix[i, j, 1] += HSIC(K, L) / num_batches
            hsic_matrix[i, j, 2] += HSIC(L, L) / num_batches

    dim = np.sqrt(hsic_matrix[:, :, 0]) * np.sqrt(hsic_matrix[:, :, 2])
    print(f"Number of zeros:{np.count_nonzero(dim==0)}")
    print(f"Index of zeros:{np.where(dim==0)}")
    print(dim)
    print(f"Number of negative1: {np.sum(hsic_matrix[:, :, 0] < 0)}")
    print(f"Number of negative2: {np.sum(hsic_matrix[:, :, 2] < 0)}")
    exit(0)
    hsic_matrix = hsic_matrix[:, :, 1] / dim
    
    
    assert not np.isnan(hsic_matrix).any(), "HSIC computation resulted in NANs"
    return hsic_matrix


    


def main():
    mkdir_if_needed(f"./{FOLDER_NAME}/")
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default='python')
    args = parser.parse_args()
    language = args.language
    if language == 'python':
        activation_folder = "activations"
        src_folder = "src_files"
    elif language == 'java':
        activation_folder = "activations_java"
        src_folder = "src_java"

    for this_model in MODEL_NAMES:
        if this_model in ['pretrained_CodeBERT']:
            print(f"Generate svg files for {this_model}")
            this_activation_name = ACTIVATION_NAMES[this_model]
            activations = load_extracted_activations(this_activation_name,activation_folder)
            print(f"Length of {this_model} activations:",len(activations))
            _, num_neurons = activations[0].shape
            for idx in range(len(activations)):
                assert activations[idx].shape[1] == num_neurons
            print(f"The number of neurons for each token in {this_model}:",num_neurons)
            hsic_matrix = cka(activations,N_SAMPLES)
            del activations

            print("-----------------------------------------------------------------")
            break

if __name__ == "__main__":
    main()


# import torch
# from torchvision.models import resnet18, resnet34, resnet50, wide_resnet50_2
# from torchvision.datasets import CIFAR10
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import numpy as np
# import random

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(0)
# np.random.seed(0)
# random.seed(0)

# model1 = resnet18(pretrained=True)
# model2 = resnet34(pretrained=True)


# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# batch_size = 256

# dataset = CIFAR10(root='../data/',
#                   train=False,
#                   download=True,
#                   transform=transform)

# dataloader = DataLoader(dataset,
#                         batch_size=batch_size,
#                         shuffle=False,
#                         worker_init_fn=seed_worker,
#                         generator=g,)

# model1 = resnet50(pretrained=True)
# model1.eval()
# model2 = wide_resnet50_2(pretrained=True)
# model2.eval()

# for (x1,*_),(x2,*_) in zip(dataloader,dataloader):
#     print(f"type(x1):{type(x1)}")
#     print(f"x1.size():{x1.size()}")
#     x1_output = model1(x1)
#     print(f"x1_output.shape:{x1_output.size()}")
#     print(f"x1_output.flatten(1):{x1_output.flatten(1).size()}")
#     exit(0)
