# this is the script I use to remove obs that have the same preceding tokens
import datetime
with open("codetest2_unique.in",'r') as f:
    code = f.readlines()
f.close()

with open("codetest2_unique.label",'r') as f:
    label = f.readlines()
f.close()

assert len(code) == len(label)

selected_token_idx = []
total = len(code)
start = datetime.datetime.now()
for idx1, this_code in enumerate(code):
    valid = False
    if this_code != "\n":
        tokens1 = this_code.split(" ")[:-1]
        idx2 = idx1 + 1
        while idx2 <= total - 1 and code[idx2]!='\n':
            tokens2 = code[idx2].split(" ")[:-1]
            if tokens2[0] != tokens1[0]:
                valid = True
            else:
                valid = False
            idx2+=1
            if not valid:
                break
        if valid:
            selected_token_idx.append(idx1)
    if (idx1+1)//1000 > 0 and (idx1+1)%1000==0:
        end = datetime.datetime.now()
        diff = end - start
        print(f"{100*(idx1+1)/total:.2f}% has been checked,{len(selected_token_idx)}\
        lines of code are saved，time passed:{end-start}")


new_code = []
new_label = []
for this_idx in selected_token_idx:
    new_code.append(code[this_idx])
    new_label.append(label[this_idx])

assert len(new_code) == len(new_label)

label_unique = []
code_unique = []
for this_code,this_label in zip(new_code,new_label):
    if this_label not in label_unique:
        label_unique.append(this_label)
        code_unique.append(this_code)

assert len(label_unique) == len(code_unique)

with open("temp.in","w") as f:
    for this_code in code_unique:
        f.writelines(f"{this_code}")
f.close()

with open("temp.label","w") as f:
    for this_label in label_unique:
        f.writelines(f"{this_label}")
f.close()
print(f"{len(code_unique)} samples are left and written to the file")
