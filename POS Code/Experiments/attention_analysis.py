from utils import extract_attentions
from run_neurox1 import MODEL_NAMES,MODEL_DESC

for this_model in MODEL_NAMES:
    if this_model in ['pretrained_BERT','pretrained_CodeBERT','pretrained_GraphCodeBERT']:
        print(f"Generating the attentions file for {this_model}")
        extract_attentions('./src_files/temp.in',MODEL_DESC[this_model])
        break
