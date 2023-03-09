#Code to randomly sampe 10% of the data from train.txt, valid.txt, test.txt and store in .word and .label files

import json
import random
import argparse

random.seed(123456)
#Creating mapping from index to function in dictionary url_to_code


def generate_balanced_dataset_files(filename):
            
    url_to_code={}
    with open('./data.jsonl') as f:
        for line in f:
            line=line.strip()
            js=json.loads(line)
            url_to_code[js['idx']]=js['func']

    data=[]
    data0=[]
    data1=[]
    with open(filename+'.txt') as f:
        for line in f:
            line=line.strip()
            url1,url2,label=line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
                if label=='0':
                    label=0
           
                elif label=='1':
                    label=1
           
                else:
                    print("corrupt sample")
           
            data.append((url1,url2,label,url_to_code))
    #stratify the data according to classes 0 and 1
            if label=='0':
                data0.append((url1,url2,label,url_to_code))
            elif label=='1':
                data1.append((url1,url2,label,url_to_code))
            else:
                print("corrupt sample")
    print(len(data0))
    print(len(data1))

    #randomly sample 10% of class 0 data
    data0=random.sample(data0,int(len(data0)*0.1))
    print(len(data0))
    #randomly sample 10% of class 1 data
    data1=random.sample(data1,int(len(data1)*0.1))
    print(len(data1))

    #balance teh dataset
    if len(data0) > len(data1):
        diff = len(data0)-len(data1)
        data0=data0[0:len(data1)]
        print(len(data0))
        print(len(data1))
    elif len(data0) < len(data1):
        data1=data1[0:len(data0)]
        print(len(data0))
        print(len(data1))

    balanced_data=data0+data1
    random.shuffle(balanced_data)
    print("balanced_data", len(balanced_data))

    #data=random.sample(data,int(len(data)*0.1))
    #print(len(data))

    #for x in range(len(data)):
    #    print(data[x][0])

    with open ('stratified/'+filename+'_url1.word', 'w', encoding='utf-8') as out:
        for x in range(len(balanced_data)):
    #        print(balanced_data[x][0])
            print(json.dumps(balanced_data[x][0]),file=out)


    with open ('stratified/'+filename+'_url2.word', 'w', encoding='utf-8') as out:
        for x in range(len(balanced_data)):
    #        print(balanced_data[x][1])
            print(json.dumps(balanced_data[x][1]),file=out)

    with open ('stratified/'+filename+'.label', 'w', encoding='utf-8') as out:
        for x in range(len(balanced_data)):
    #        print(balanced_data[x][2])
            print(json.dumps(balanced_data[x][2]),file=out)


def main():
    # Create the parser and add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='filename',type=str, help="train,valid or test")

    # Parse and print the results
    args = parser.parse_args()
    print(args.filename)
    generate_balanced_dataset_files(args.filename)


if __name__ == "__main__":
    main()


'''
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

'''



  
   
