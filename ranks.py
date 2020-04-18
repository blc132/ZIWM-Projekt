import pandas as pd
import numpy as np
import os
import glob

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

script_dir = os.path.dirname(__file__)

def get_data(files):
    data_matrix = np.loadtxt(files[0], dtype='i', delimiter='\t')
    data_matrix = data_matrix.T

    last_col = [0] * len(data_matrix)
    data_matrix = np.column_stack((data_matrix, last_col))

    for x in range(len(files) -1):
        temp_matrix = np.loadtxt(files[x + 1], dtype='i', delimiter='\t')
        temp_matrix = temp_matrix.T
        last_col = [x+1] * len(temp_matrix)
        temp_matrix = np.column_stack((temp_matrix, last_col))
        data_matrix = np.concatenate((data_matrix, temp_matrix), axis=0)
        
    X  = data_matrix[:, :-1]
    Y = data_matrix[:,-1]
    return X, Y



def main():
    files = glob.glob('sets/*.txt')
    X, Y = get_data(files)

    fvalue_selector = SelectKBest(f_classif)
    fvalue_selector.fit(X, Y)

    rank = fvalue_selector.scores_

    indexes = rank.argsort()[-10:][::-1]

    for index in indexes:
        print(f'{index + 1}: {rank[index]}')

if __name__ == "__main__":
    main()


