import os

f = open('code.txt','r')
code = f.readlines()
f.close()
f = open('label.txt','r')
label = f.readlines()
f.close()

assert len(code) == len(label)

# drop observations that the length of code does not
# equal to the length of labels
dropped_idx = 0
f = open('codetest2.in','w')
g = open('codetest2.label','w')

for this_code, this_label in zip(code,label):
    if len(this_code.split(" ")) != len(this_label.split(" ")):
        dropped_idx+=1
        continue
    # elif max([len(i) for i in this_code.split(" ")]) > 512:
    elif len(this_code.split(" ")) > 512:
        dropped_idx+=1
        continue
    else:
        f.writelines(this_code)
        g.writelines(this_label)
f.close()
g.close()

# SANITY CHECK: ensure the # of observations in both files are the same.
with open("codetest2.in",'r') as f:
    code = f.readlines()
f.close()

with open("codetest2.label",'r') as f:
    label = f.readlines()
f.close()

assert len(code) == len(label)

OUTPUT_IN = "codetest2_test_unique.in"
OUTPUT_LABEL = "codetest2_test_unique.label"
FOLDER = './'

code_unique = []
label_unique = []
for this_code, this_label in zip(code, label):
    if this_code not in code_unique:
        code_unique.append(this_code)
        label_unique.append(this_label)

assert len(code_unique) == len(label_unique)
with open(os.path.join(FOLDER, OUTPUT_IN),"w") as f:
    for this_code in code_unique:
        f.writelines(f"{this_code}")
f.close()

with open(os.path.join(FOLDER, OUTPUT_LABEL),"w") as f:
    for this_label in label_unique:
        f.writelines(f"{this_label}")
f.close()
print(f"After removing redundant observations, {len(code_unique)} samples are\
left and written to the {OUTPUT_IN} and {OUTPUT_LABEL} files")
