import json
import random

random.seed(123456)
#Creating mapping from index to function in dictionary url_to_code
url_to_code={}
with open('./data.jsonl') as f:
    for line in f:
        line=line.strip()
        js=json.loads(line)
        url_to_code[js['idx']]=js['func']
print(url_to_code)

data=[]
with open('train.txt') as f:
    count0=0
    count1 =0
    for line in f:
        line=line.strip()
        url1,url2,label=line.split('\t')
        if url1 not in url_to_code or url2 not in url_to_code:
            continue
            if label=='0':
                label=0
                count0=count0+1
            else:
               label=1
               count1=count1+1
        data.append((url1,url2,label,url_to_code))
    print("Total train set- 0:", count0, "1:", count1)
print(data)
#
data=[]
with open('valid.txt') as f:
    count0=0
    count1 =0
    for line in f:
        line=line.strip()
        url1,url2,label=line.split('\t')
        if url1 not in url_to_code or url2 not in url_to_code:
            continue
            if label=='0':
                label=0
                count0=count0+1
            else:
               label=1
               count1=count1+1
        data.append((url1,url2,label,url_to_code))
    print("Total validation set- 0:", count0, "1:", count1)

data=[]
with open('test.txt') as f:
    count0=0
    count1 =0
    for line in f:
        line=line.strip()
        url1,url2,label=line.split('\t')
        if url1 not in url_to_code or url2 not in url_to_code:
            continue
            if label=='0':
                label=0
                count0=count0+1
            else:
               label=1
               count1=count1+1
        data.append((url1,url2,label,url_to_code))
    print("Total test set- 0:", count0, "1:", count1)


#with open('train.label', encoding='utf-8') as j:
#    count0 = 0
#    count1=0
#    for line in j:
#        d = line.replace('"', '').replace("'", "")
#        #print(d)
#        if"0"in d:
#            count0=count0+1
#            continue
#        elif "1" in d:
#            count1=count1+1 
#            continue
#        else:
#            print("error:train")
#    print("Train dataset- 0:", count0,"1:", count1)
#
#with open('dev.label', encoding='utf-8') as j:
#    count0 = 0
#    count1=0
#    for line in j:
#        d = line.replace('"', '').replace("'", "")
#        
#        #print(d)
#        if "0" in d:
#            count0=count0+1
#            continue
#        elif "1":
#            count1=count1+1
#            continue
#        else:
#            print("error:dev")
#    print("Validation dataset- 0:", count0,"1:", count1)
#
#with open('test.txt', encoding='utf-8') as j:
#    count0 = 0
#    count1=0
#    for line in j:
#        d =line.split()[2]
#        #print(d)
#        if d=="0":
#            count0=count0+1
#            continue
#        elif d=="1":
#            count1=count1+1
#            continue
#        else:
#            print("error")
#    print("Test dataset- 0:", count0,"1:", count1)
