import re
import csv

with open('skip_logs') as f_in:
    lst = []
    for line in f_in:
        grp = re.search(r"Skipping line:\s+([0-9]+)",line)
        #Skip if None; eg 23-27
        if grp is None:
            continue 
        x = grp.groups() #returns tuples
        #print(x)
        lst.append(int(x[0])+1)
    print(lst)

line_num = 0
with open('codetest.in') as f_in, open('codetest2.in','w') as f_out:
    for line in f_in.readlines():
        line_num = line_num + 1
        if line_num in lst:
            continue
        else:
            f_out.writelines(line)

line_num_l = 0
with open ('codetest.label') as f_label, open('codetest2.label','w') as f_outl:
    for line in f_label.readlines():
        line_num_l = line_num_l + 1
        if line_num_l in lst:
            continue
        else:
            f_outl.writelines(line)
