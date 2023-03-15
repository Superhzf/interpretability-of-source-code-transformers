
Clone Detection Dataset statistics (stratified sampling)
[arushi17@pronto dataset]$ python preprocess.py --filename=train --stats
train
450166
450862
45016
45086
45016
45016
balanced_data 90032
train dataset- 0: 45016 1: 45016
[arushi17@pronto dataset]$ ls stratified
train.label  train_url1.word  train_url2.word
[arushi17@pronto dataset]$ python preprocess.py --filename=valid --stats
valid
361577
53839
36157
5383
5383
5383
balanced_data 10766
valid dataset- 0: 5383 1: 5383
[arushi17@pronto dataset]$ python preprocess.py --filename=test --stats
test
358596
56820
35859
5682
5682
5682
balanced_data 11364
test dataset- 0: 5682 1: 5682
