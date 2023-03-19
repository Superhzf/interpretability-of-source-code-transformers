from utils import extract_attentions
from run_neurox1 import MODEL_NAMES,MODEL_DESC

for this_model in MODEL_NAMES:
    if this_model in ['pretrained_GPT2','pretrained_BERT']:
        print(f"Generating the attentions file for {this_model}")
        extract_attentions(MODEL_DESC[this_model],'./src_files/temp.in',)
        break
