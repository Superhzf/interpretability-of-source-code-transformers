import os
import pandas as pd


files = pd.read_pickle(r'deduplicated_python_code.pickle')
# files = files[:5000] # Train
# files = files[5000:8000] # valid
files = files[8000:11000] # Test

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
