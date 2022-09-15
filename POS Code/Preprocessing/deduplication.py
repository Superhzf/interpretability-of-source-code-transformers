with open("codetest2.in",'r') as f:
    code = f.readlines()
f.close()

with open("codetest2.label",'r') as f:
    label = f.readlines()
f.close()

assert len(code) == len(label)

code_unique = []
label_unique = []
for this_code, this_label in zip(code, label):
    if this_code not in code_unique:
        code_unique.append(this_code)
        label_unique.append(this_label)

assert len(code_unique) == len(label_unique)

with open("codetest2_unique.in","w") as f:
    for this_code in code_unique:
        f.writelines(f"{this_code}")
f.close()

with open("codetest2_unique.label","w") as f:
    for this_label in label_unique:
        f.writelines(f"{this_label}")
f.close()
print(f"After removing redundant observations, {len(code_unique)} samples are\
left and written to the codetest2_unique.in and codetest2_unique.label files")