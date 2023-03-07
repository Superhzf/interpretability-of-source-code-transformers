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

data=[]
data0=[]
data1=[]
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
                data0.append((url1,url2,label,url_to_code))
            else:
               label=1
               count1=count1+1
               data1.append((url1,url2,label,url_to_code))
        data.append((url1,url2,label,url_to_code))
    print("Total train set- 0:", count0, "1:", count1)

#Randomly sample 10% of teh data
data0=random.sample(data0,int(len(data0)*0.1))
data1=random.sample(data1,int(len(data1)*0.1))
print(len(data0))
print(len(data1))

for x in range(len(data0)):
    print(data0[x][2])

for x in range(len(data1)):
    print(data1[x][2])


#
#with open ('train0.word', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][0])
#        print(json.dumps(data[x][0]),file=out)
#
#
#with open ('train1.word', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][1])
#        print(json.dumps(data[x][1]),file=out)
#
#with open ('train.label', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][2])
#        print(json.dumps(data[x][2]),file=out)
#
#data=[]
#with open('valid.txt') as f:
#    for line in f:
#        line=line.strip()
#        url1,url2,label=line.split('\t')
#        if url1 not in url_to_code or url2 not in url_to_code:
#            continue
#            if label=='0':
#                label=0
#            else:
#               label=1
#        data.append((url1,url2,label,url_to_code))
#data=random.sample(data,int(len(data)*0.1))
#
#print(len(data))
#
#with open ('dev0.word', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][0])
#        print(json.dumps(data[x][0]),file=out)
#
#
#with open ('dev1.word', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][1])
#        print(json.dumps(data[x][1]),file=out)
#
#with open ('dev.label', 'w', encoding='utf-8') as out:
#    for x in range(len(data)):
#        print(data[x][2])
#        print(json.dumps(data[x][2]),file=out)
#






         
