import glob
import json
import pandas as pd
import os
import re

def save(original_file,new_file_path,new_file_name):
    f = open(original_file,'r')
    lines = f.readlines()
    f.close()
    path = os.path.join(new_file_path, new_file_name)
    path1 = f"{path}_func.txt"
    path2 = f"{path}_target_mask.txt"
    path3 = f"{path}_error_location_mask.txt"
    path4 = f"{path}_candidate_mask.txt"
    with open(path1,'a') as f1, open(path2,'a') as f2,open(path3,'a') as f3,open(path4,'a') as f4:
        for this_line in lines:
            this_line = json.loads(this_line)
            this_func = this_line['function']
            this_target_mask = this_line['target_mask']
            this_error_location_mask = this_line['error_location_mask']
            this_candidate_mask = this_line['candidate_mask']

            assert len(this_func) == len(this_target_mask)
            assert len(this_target_mask) == len(this_error_location_mask)
            assert len(this_error_location_mask) == len(this_candidate_mask)

            this_candidate_mask = [str(this) for this in this_candidate_mask]
            this_candidate_mask = ' '.join(this_candidate_mask)

            this_target_mask = [str(this) for this in this_target_mask]
            this_target_mask = ' '.join(this_target_mask)

            this_error_location_mask = [str(this) for this in this_error_location_mask]
            this_error_location_mask = ' '.join(this_error_location_mask)

            this_func = ' '.join(this_func)
            f1.write(f"{this_func}\n")
            f2.write(f"{this_target_mask}\n")
            f3.write(f"{this_error_location_mask}\n")
            f4.write(f"{this_candidate_mask}\n")

        f1.close()
        f2.close()
        f3.close()
        f4.close()


def main(original_file_folder,new_file_folder):
    trainPath=os.path.join(original_file_folder, "*train*")
    devPath=os.path.join(original_file_folder, "*dev*")
    evalPath=os.path.join(original_file_folder, "*eval*")
    trainList = glob.glob(trainPath)
    devList = glob.glob(devPath)
    evalList = glob.glob(evalPath)
    for idx,thisList in enumerate([trainList,devList,evalList]):
        file_name = ['train','dev','eval'][idx]
        path = os.path.join(new_file_folder, file_name)
        path1 = f"{path}_func.txt"
        path2 = f"{path}_target_mask.txt"
        path3 = f"{path}_error_location_mask.txt"
        path4 = f"{path}_candidate_mask.txt"
        f1 = open(path1,'w')
        f2 = open(path2,'w')
        f3 = open(path3,'w')
        f4 = open(path4,'w')
        for thisFile in thisList:
            save(thisFile,new_file_folder,file_name)


if __name__ == '__main__':
    original_file_path='./original/'
    new_file_path='./generated/'
    main(original_file_path,new_file_path)
