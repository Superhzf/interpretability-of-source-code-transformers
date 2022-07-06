import glob
import json
import pandas as pd
import os
import re

def save(original_file,new_file_path_list,idx):
    path_csv, path_func, path_docstr, path_label = new_file_path_list
    # save as CSV
    this_df = pd.read_json(original_file,lines=True)
    this_df.drop(["info"],inplace=True,axis=1)
    this_df.function = this_df.function.apply(lambda x : x.replace('\n', '\\n'))
    this_df.function = this_df.function.apply(lambda x : x.replace('\t', '\\t'))
    this_df.function = this_df.function.apply(lambda x : x.replace('\r', '\\r'))
    this_df.docstring = this_df.docstring.apply(lambda x : x.replace('\n', '\\n'))
    this_df.docstring = this_df.docstring.apply(lambda x : x.replace('\t', '\\t'))
    this_df.docstring = this_df.docstring.apply(lambda x : x.replace('\r', '\\r'))
    this_df.label = this_df.label.apply(lambda x : int(x.lower()=="correct"))

    count = this_df.function.str.len().values
    # 32700 is the maximum # characters a CSV cell can contain.
    this_df = this_df.loc[count <= 32700]
    num_ignored = len(count) - len(this_df)
    if num_ignored>0:
        print(f"{num_ignored} observations are ignored while generating {path_csv} \
            because of too many characters")
    if idx == 0:
        this_df.to_csv(path_csv,index=False,mode='a')
    else:
        this_df.to_csv(path_csv,index=False,mode='a',header=False)

    #Save as TXT
    f = open(original_file,'r')
    lines = f.readlines()
    f.close()
    count_txt = 0
    assert len(count) == len(lines), "# of obs from DataFrame should equal # of obse from TXT"
    with open(path_func, 'a') as f1, open(path_docstr, 'a') as f2, open(path_label,'a') as f3:
        for this_line, this_count in zip(lines,count):
            if this_count<=32700:
                this_line = json.loads(this_line)
                this_func = this_line['function']
                this_docstr = this_line['docstring']
                this_label = this_line['label']

                this_func = this_func.replace('\n','\\n')
                this_func = this_func.replace('\t','\\t')
                this_func = this_func.replace('\r','\\r')

                this_docstr = this_docstr.replace('\n','\\n')
                this_docstr = this_docstr.replace('\t','\\t')
                this_docstr = this_docstr.replace('\r','\\r')

                this_label = int(this_label.lower() == 'correct')

                f1.write(f"{this_func}\n")
                f2.write(f"{this_docstr}\n")
                f3.write(f"{this_label}\n")
                count_txt += 1

        f1.close()
        f2.close()
        f3.close()
        assert count_txt == len(this_df), "# of obs written to CSV should equal \
            to the # of obsa written to TXT file."


def main(original_file_folder,new_file_folder):
    trainPath=os.path.join(original_file_folder, "*train*")
    devPath=os.path.join(original_file_folder, "*dev*")
    evalPath=os.path.join(original_file_folder, "*eval*")
    trainList = glob.glob(trainPath)
    devList = glob.glob(devPath)
    evalList = glob.glob(evalPath)
    for idx1,thisList in enumerate([trainList,devList,evalList]):
        file_name = ['train','dev','eval'][idx1]
        path = os.path.join(new_file_path, file_name)
        path_csv = f"{path}.csv"
        path_func = f"{path}_func.txt"
        path_docstr = f"{path}_docstring.txt"
        path_label = f"{path}_label.txt"
        path_list = [path_csv,path_func,path_docstr,path_label]
        # Initialize files
        for this_path in path_list:
            f = open(this_path,'w')
            f.close()
        for idx2, thisFile in enumerate(thisList):
            save(thisFile,path_list,idx2)


if __name__ == '__main__':
    original_file_path='./original/'
    new_file_path='./generated/'
    main(original_file_path,new_file_path)
    print("All done!")
