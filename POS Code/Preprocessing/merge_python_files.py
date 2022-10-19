import os


folder = "./py150_files/"
file_name = "train.txt"
f = open(os.path.join(folder,file_name),'r')
# this way will remove the newline.
eval_files = f.read().splitlines()
f.close()

my_file = "myfile.txt"

try:
    os.remove(my_file)
except OSError:
    pass

g = open(my_file,'a')
for idx,this_file_name in enumerate(eval_files):
    complete_file = os.path.join(folder,this_file_name)
    f = open(complete_file,'r')
    py_code = f.readlines()
    f.close()
    g.writelines(py_code)
    if idx == 500:
        break
g.close()
