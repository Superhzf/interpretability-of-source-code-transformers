import os
import pandas as pd

# folder1 = "./"
# folder2 = "./py150_files/"
# file_name = "test.txt"
# f = open(os.path.join(folder1,file_name),'r')
# # this way will remove the newline.
# eval_files = f.read().splitlines()
# f.close()

# my_file = "myfile.txt"

# try:
#     os.remove(my_file)
# except OSError:
#     pass

# g = open(my_file,'a')
# for idx,this_file_name in enumerate(eval_files):
#     complete_file = os.path.join(folder2,this_file_name)
#     f = open(complete_file,'r')
#     py_code = f.readlines()
#     f.close()
#     g.writelines(py_code)
    
# g.close()



# folder1 = "./"
# folder2 = "./py150_files/"
# file_name = "test.txt"
# f = open(os.path.join(folder1,file_name),'r')
# # this way will remove the newline.
# eval_files = f.read().splitlines()
# f.close()

files = pd.read_pickle(r'deduplicated_python_code.pickle')
files = files[:6000] # Train
# files = files[6000:8000] # Dev
# files = files[8000:10000] # Test

my_file = "myfile.txt"

try:
    os.remove(my_file)
except OSError:
    pass

g = open(my_file,'a')
for this_file_py_code in files:
    g.writelines(this_file_py_code)
    g.writelines("\n")
    g.writelines('\n')
    
g.close()
