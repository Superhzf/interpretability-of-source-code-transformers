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


FOLDER = './'

code_unique = []
label_unique = []
for this_code, this_label in zip(code, label):
    if this_code not in code_unique:
        code_unique.append(this_code)
        label_unique.append(this_label)

assert len(code_unique) == len(label_unique)

per_train = 0.5
OUTPUT_IN_TRAIN = "codetest2_train_unique.in"
OUTPUT_LABEL_TRAIN = "codetest2_train_unique.label"
per_valid = 0.25
OUTPUT_IN_VALID = "codetest2_valid_unique.in"
OUTPUT_LABEL_VALID = "codetest2_valid_unique.label"
per_test = 0.25
OUTPUT_IN_TEST = "codetest2_test_unique.in"
OUTPUT_LABEL_TEST = "codetest2_test_unique.label"

per = [per_train,per_valid,per_test]
files = [(OUTPUT_IN_TRAIN,OUTPUT_LABEL_TRAIN),(OUTPUT_IN_VALID,OUTPUT_LABEL_VALID),(OUTPUT_IN_TEST,OUTPUT_LABEL_TEST)]

count = 0
for idx,combo in enumerate(zip(per,files)):
    this_per, this_file = combo
    if idx == 0:
        end = int(len(code_unique) * this_per)
        this_code = code_unique[:end]
        this_label = label_unique[:end]
    elif idx != len(per) - 1:
        start = int(len(code_unique) * this_per)
        end = int(len(code_unique) * per[idx+1])
        this_code = code_unique[start:end]
        this_code = label_unique[start:end]
    else:
        start = int(len(code_unique) * this_per)
        this_code = code_unique[start:]
        this_label = label_unique[start:]

    count += len(this_label)
    with open(os.path.join(FOLDER, files[0]),"w") as f:
        for this_code in code_unique:
            f.writelines(f"{this_code}")
    f.close()

    with open(os.path.join(FOLDER, files[1]),"w") as f:
        for this_label in label_unique:
            f.writelines(f"{this_label}")
    f.close()

print(f"After removing redundant observations, {count} samples are\
left and written to files")
