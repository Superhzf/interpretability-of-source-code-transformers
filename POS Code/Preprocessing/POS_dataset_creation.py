import os
import re

''' Create codetest.in and codetest.label from python code file myfile.txt for word level analysis'''

try:
    os.remove("myfile_tokens.txt")
except OSError:
    pass

#Run python tokenizer on myfile.txt -- which contains original code file
os.system("python -m tokenize -e myfile.txt > myfile_tokens.txt")

keyword_list = ['False','await','else','import','pass','None','break','except','in','raise','True','class','finally','is','return','and','continue','for','lambda','try','as','def','from','nonlocal','while','assert','del','global','not','with','async''elif','if','or','yield']

#Creating dictionary of tokens and tags from original python source code
with open('myfile_tokens.txt') as f_in:
    lst = []
    for line in f_in:
        grp = re.search(r"([0-9]+,[0-9]+-[0-9]+,[0-9]+:)\s+([A-Z]+)\s+('.*')\s*|[0-9]+,[0-9]+-[0-9]+,[0-9]+:\s+[A-Z]+\s+'.*'\s*",line)
        #Skip if None; eg 23-27
        if grp is None:
            continue
        x = grp.groups() #returns tuples
        my_list = [x[1], x[2][1:-1]] #convert to mutable lists / strip quotes fro x[2]
        #custom tokens for indent and dedent, identifying keywords
        for item in (my_list):
            if my_list[0] == 'INDENT':
                my_list[1] = '~~~'
            if my_list[0] == 'DEDENT':
                my_list[1] = '~~'
            if ((my_list[0] == 'NAME') and (my_list[1] in keyword_list)):
                my_list[0] = 'KEYWORD'
        lst.append(my_list) #store in list of lists
        clean = [x for x in lst if x is not None]
        tup_to_dict = dict(clean)

f_in.close()

# try:
#     os.remove("POS.csv")
# except OSError:
#     pass

# #Store intermediate result in csv file
# with open('POS.csv', 'a') as f:
#     for item in clean:
#         f.write("%s,%s\n"%(item[0],item[1]))
# f.close()

try:
    os.remove('codetest.label')
except OSError:
    pass

try:
    os.remove('codetest.in')
except OSError:
    pass

#Convert to required input formats: tokens: codetest.in, tags: codetest.label
with open('codetest.label', 'a') as f_label, open('codetest.in', 'a') as f_in:
  for i in range(3000):
    for i in range(len(lst)):
      if (lst[i-1][0] == 'NEWLINE' or lst[i-1][0] == 'NL'):
        # print("\n")
        f_label.writelines("\n")
      if (lst[i][0] == 'ENDMARKER'):
        continue
      else:
        # print(lst[i][0], end = ' ')
        f_label.writelines(lst[i][0] + ' ')

    for i in range(len(lst)):
      if (lst[i-1][0] == 'NEWLINE' or lst[i-1][0] == 'NL'):
        # print("\n")
        f_in.writelines("\n")
      if (lst[i][0] == 'ENDMARKER'):
        # print("\n")
        # f_in.writelines("\n")
        continue
      else:
        # print(lst[i][1], end = ' ')
        f_in.writelines(lst[i][1]+ ' ' )

f_label.close()
f_in.close()
