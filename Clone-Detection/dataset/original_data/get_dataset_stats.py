import json


with open('train.label', encoding='utf-8') as j:
    count0 = 0
    count1=0
    for line in j:
        d = line.replace('"', '').replace("'", "")
        #print(d)
        if"0"in d:
            count0=count0+1
            print("line 0", line)
            continue
        elif "1" in d:
            count1=count1+1 
            print("line 1", line)
            continue
        else:
            print("error:train")
    print("Train dataset- 0:", count0,"1:", count1)

with open('dev.label', encoding='utf-8') as j:
    count0 = 0
    count1=0
    for line in j:
        d = line.replace('"', '').replace("'", "")
        
        #print(d)
        if "0" in d:
            count0=count0+1
            continue
        elif "1":
            count1=count1+1
            continue
        else:
            print("error:dev")
    print("Validation dataset- 0:", count0,"1:", count1)

with open('test.txt', encoding='utf-8') as j:
    count0 = 0
    count1=0
    for line in j:
        d =line.split()[2]
        #print(d)
        if d=="0":
            count0=count0+1
            continue
        elif d=="1":
            count1=count1+1
            continue
        else:
            print("error")
    print("Test dataset- 0:", count0,"1:", count1)
