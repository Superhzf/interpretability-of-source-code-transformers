import os
FOLDER = './'
INPUT_IN = "codetest2.in"
INPUT_LABEL = "codetest2.label"
OUTPUT_IN = "codetest2_test_unique.in"
OUTPUT_LABEL = "codetest2_test_unique.label"

with open(os.path.join(FOLDER, INPUT_IN),'r') as f:
    code = f.readlines()
f.close()

with open(os.path.join(FOLDER, INPUT_LABEL),'r') as f:
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
