import torch
import time
from sklearn.utils import shuffle
import numpy as np
import sklearn
###sklearn version
print('The scikit-learn version is {}.'.format(sklearn.__version__))

###CPU
start_time = time.time()
a = torch.ones(400,400)
for _ in range(100):
    a += a
elapsed_time = time.time() - start_time

print('CPU time = ',elapsed_time)

###GPU
start_time = time.time()
b = torch.ones(400,400).cuda()
for _ in range(100):
    b += b
elapsed_time = time.time() - start_time

print('GPU time = ',elapsed_time)

X = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
y = np.array([0, 1, 2, 3, 4])
X, y = shuffle(X, y,random_state=50)
print(X)
print(y)
