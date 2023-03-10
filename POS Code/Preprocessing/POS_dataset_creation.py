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

try:
    os.remove('codetest.label')
except OSError:
    pass

try:
    os.remove('codetest.in')
except OSError:
    pass

#Creating dictionary of tokens and tags from original python source code
with open('codetest.label', 'a') as f_label, open('codetest.in', 'a') as f_in:
  with open('myfile_tokens.txt') as f_token:
      for line in f_token:
        grp1 = re.search(r"""([0-9]+,[0-9]+-[0-9]+,[0-9]+:)\s*([A-Z]+)\s*[']{1}(.*?)[']{1}""",line)
        grp2 = re.search(r"""([0-9]+,[0-9]+-[0-9]+,[0-9]+:)\s*([A-Z]+)\s*["]{1}(.*?)["]{1}""",line)
        if grp1 is not None and grp2 is not None:
            if len(grp1[2])>len(grp2[2]):
                grp = grp1
            else:
                grp = grp2
        elif grp1 is not None:
            grp = grp1
        elif grp2 is not None:
            grp = grp2
        else:
            print(f"Something is wrong with, {i}, {line}")
        x = grp.groups()
        my_list = [x[1], x[2]]
        for item in (my_list):
            if my_list[0] == 'INDENT':
                my_list[1] = '~~~'
            elif my_list[0] == 'DEDENT':
                my_list[1] = '~~'
            elif ((my_list[0] == 'NAME') and (my_list[1] in keyword_list)):
                my_list[0] = 'KEYWORD'

        if (my_list[0] == 'NEWLINE' or my_list[0] == 'NL'):
            f_label.writelines("\n")
            f_in.writelines("\n")
        elif (my_list[0] == 'ENDMARKER'):
            continue
        else:
            f_label.writelines(my_list[0] + ' ')
            f_in.writelines(my_list[1]+ ' ' )
    


f_in.close()
f_label.close()
f_in.close()
