import json

with open ('train.label', 'w', encoding='utf-8') as out:
    with open('train.jsonl', encoding='utf-8') as j:
        for line in j:
            d = json.loads(line)['target']
            print(json.dumps(d), file=out)

with open ('train.word', 'w', encoding='utf-8') as out:
    with open('train.jsonl', encoding='utf-8') as j:
        for line in j:
            d = json.loads(line)['func']
            print(json.dumps(d), file=out)


import json

with open ('dev.label', 'w', encoding='utf-8') as out:
    with open('valid.jsonl', encoding='utf-8') as j:
        for line in j:
            d = json.loads(line)['target']
            print(json.dumps(d), file=out)

with open ('dev.word', 'w', encoding='utf-8') as out:
    with open('valid.jsonl', encoding='utf-8') as j:
        for line in j:
            d = json.loads(line)['func']
            print(json.dumps(d), file=out)
