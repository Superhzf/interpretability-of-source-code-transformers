# Convert dataset from the [cuBERT project](https://github.com/mhagglun/google-research-cuBERT) to the [GLUE](https://huggingface.co/datasets/glue) format



## Usage:

1. Download datasets from [here](https://github.com/mhagglun/google-research-cuBERT#benchmarks-and-fine-tuned-models), which includes 6 experiments. The datasets should be placed in experiment_subfolder/original/
2. In each experiment subfolder, run `python cubert2glue*.py`

## Expected result:

* For task1, task2, task3, task4, and task5, you are expected to see both *.csv files and *.txt files. For task 6, only *.txt files are generated because in task 6 the type all observations is list, and the length varies.

* Since the maximum number of characters that a CSV cell can contain is ~32700, any observations that include more than that will be ignored.

* In cuBERT, the training set may include multiple files, whereas here I combine them into one train file. Similar things work for dev and eval.
