import pandas as pd
import numpy as np
import os

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

script_dir = os.path.dirname(__file__)
files_dir = os.path.join(script_dir, 'sets')

files = os.listdir(files_dir)

data_matrix = np.loadtxt(f"{files_dir}/{files[0]}", dtype='i', delimiter='\t')
data_matrix = data_matrix.T

last_col = [0] * len(data_matrix)
data_matrix = np.column_stack((data_matrix, last_col))

for x in range(len(files) -1):
    temp_matrix = np.loadtxt(f"{files_dir}/{files[x+1]}", dtype='i', delimiter='\t')
    temp_matrix = temp_matrix.T
    last_col = [x+1] * len(temp_matrix)
    temp_matrix = np.column_stack((temp_matrix, last_col))
    data_matrix = np.concatenate((data_matrix, temp_matrix), axis=0)

np.savetxt('text.txt',data_matrix,fmt='%i')
X  = data_matrix[:, :-1]
Y = data_matrix[:,-1]

fvalue_selector = SelectKBest(f_classif)
fvalue_selector.fit(X, Y)

rank = fvalue_selector.scores_

indexes = rank.argsort()[-10:][::-1]

for index in indexes:
    print(f'{index}: {rank[index]}')