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
with open('train.txt') as f:
    for line in f:
        line=line.strip()
        url1,url2,label=line.split('\t')
        if url1 not in url_to_code or url2 not in url_to_code:
            continue
            if label=='0':
                label=0
            else:
               label=1
        data.append((url1,url2,label,url_to_code))
data=random.sample(data,int(len(data)*0.1))

print(len(data))
for x in range(len(data)):
    print(data[x][0])

with open ('train.word', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][0])
        print(json.dumps(data[x][0]),file=out)


with open ('train1.word', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][1])
        print(json.dumps(data[x][1]),file=out)

with open ('train.label', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][2])
        print(json.dumps(data[x][2]),file=out)

data=[]
with open('valid.txt') as f:
    for line in f:
        line=line.strip()
        url1,url2,label=line.split('\t')
        if url1 not in url_to_code or url2 not in url_to_code:
            continue
            if label=='0':
                label=0
            else:
               label=1
        data.append((url1,url2,label,url_to_code))
data=random.sample(data,int(len(data)*0.1))

print(len(data))

with open ('dev.word', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][0])
        print(json.dumps(data[x][0]),file=out)


with open ('dev1.word', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][1])
        print(json.dumps(data[x][1]),file=out)

with open ('dev.label', 'w', encoding='utf-8') as out:
    for x in range(len(data)):
        print(data[x][2])
        print(json.dumps(data[x][2]),file=out)




#with open ('train1.word', 'w', encoding='utf-8') as out:
#    with open('train.txt') as f:
#        for line in f:
#            line=line.strip()
#            url1,url2,label=line.split('\t')
#            if url1 not in url_to_code or url2 not in url_to_code:
#                continue
#                if label=='0':
#                    label=0
#                else:
#                    label=1
#            print(json.dumps(url2),file=out)
#           
#with open ('train.label', 'w', encoding='utf-8') as out:
#    with open('train.txt') as f:
#        for line in f:
#            line=line.strip()
#            url1,url2,label=line.split('\t')
#            if url1 not in url_to_code or url2 not in url_to_code:
#                continue
#                if label=='0':
#                    label=0
#                else:
#                    label=1
#            print(json.dumps(label),file=out)
#
#
#
#data=[]
#with open ('dev.word', 'w', encoding='utf-8') as out:
#    with open('valid.txt') as f:
#        for line in f:
#            line=line.strip()
#            url1,url2,label=line.split('\t')
#            if url1 not in url_to_code or url2 not in url_to_code:
#                continue
#                if label=='0':
#                    label=0
#                else:
#                    label=1
#            print(json.dumps(url1),file=out)
#            data.append((url1,url2,label,url_to_code))
#
#with open ('dev1.word', 'w', encoding='utf-8') as out:
#    with open('valid.txt') as f:
#        for line in f:
#            line=line.strip()
#            url1,url2,label=line.split('\t')
#            if url1 not in url_to_code or url2 not in url_to_code:
#                continue
#                if label=='0':
#                    label=0
#                else:
#                    label=1
#            print(json.dumps(url2),file=out)
#
#with open ('dev.label', 'w', encoding='utf-8') as out:
#    with open('valid.txt') as f:
#        for line in f:
#            line=line.strip()
#            url1,url2,label=line.split('\t')
#            if url1 not in url_to_code or url2 not in url_to_code:
#                continue
#                if label=='0':
#                    label=0
#                else:
#                    label=1
#            print(json.dumps(label),file=out)
#




         
