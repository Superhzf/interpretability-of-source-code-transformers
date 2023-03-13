data.jsonl contained original dataset i.e CDLH (BigCloneBench) with 9126 code samples total which can be used to make 1,731,857 pairs. train.txt, valid.txt,test.txt contain labelled pairs of code snippets/clones from data.json.The ratio of the split was 52/24/24. They contain 901028, 415416, 415416 samples respectively. The pretrained model is finetuned using 10% of samples from the train and validation sets. These samples were randomly selected. The train.word, train.label and dev.word, dev.label files were created by also randomly sampling 10% of the data with the same seed 123456.This is also the seed used by the CodeXGLUE implementation. The code for generating the .word and .label files can be found in preprocess.py These files contain 90102,41541 samples for the train and dev sets respectively.
The class distribution for these datasets is as follows.
Train dataset- 0: 45,216 , 1: 44,886
Validation dataset- 0: 36,085 1: 5,456
Test dataset- 0: 358,596 1: 56,820  (Original size, not 10%)

We observe a major class imbalance in the validation and test sets, with the number of true clone samples being significantly lower than false ones. 

Instead of using random sampling, we now use stratified sampling to address class imbalance.We save the stratified datasets we use as well. 
