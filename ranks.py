import pandas as pd
import numpy as np
import os
import glob

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif, chi2
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from copy import deepcopy

script_dir = os.path.dirname(__file__)


def get_data(files):
    data_matrix = np.loadtxt(files[0], dtype='i', delimiter='\t')
    data_matrix = data_matrix.T

    last_col = [0] * len(data_matrix)
    data_matrix = np.column_stack((data_matrix, last_col))

    for x in range(len(files) - 1):
        temp_matrix = np.loadtxt(files[x + 1], dtype='i', delimiter='\t')
        temp_matrix = temp_matrix.T
        last_col = [x+1] * len(temp_matrix)
        temp_matrix = np.column_stack((temp_matrix, last_col))
        data_matrix = np.concatenate((data_matrix, temp_matrix), axis=0)

    X = data_matrix[:, :-1]
    Y = data_matrix[:, -1]
    return X, Y


def classification(X, Y, activation_value, momentum_value, top_rank, fvalue_selector, layer_size, best_score, list_score):

    # Utworzenie walidacji krzyżowej, podział na 2 grupy, 5 powtórzeń
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5)

    # Utworzenie modelu o wcześniej zdefiniowanych parametrach
    mlp = MLPClassifier(hidden_layer_sizes=layer_size,
                        activation=activation_value, max_iter=200, momentum=momentum_value)

    first_time = True
    # Pętla dla 5 krotnej walidacji krzyżowej
    for train, test in rkf.split(X, Y):
        x_train, x_test = X[train], X[test]
        y_train, y_test = Y[train], Y[test]

        # Dopasowanie x_train do y_train
        mlp.fit(x_train, y_train)

        # Przypisanie wyniku na podstawie nowych zbiorów
        score = mlp.score(x_test, y_test)

        # Przewidywanie za pomocą wcześniej utworzonego mlp przy użyciu zbioru x_test
        predict = mlp.predict(x_test)

        # Macierz pomyłek
        confusion_matrix_value = confusion_matrix(y_test, predict)

        # Utworzenie nowego rekordu z danymi odnośnie wykonanego badania
        if first_time:
            temp_score = score
            temp_confusion_matrix = confusion_matrix_value
            first_time = False
        else:
            temp_score = temp_score + score
            temp_confusion_matrix = temp_confusion_matrix + confusion_matrix_value

    current_score = [temp_score/5, activation_value, momentum_value,
                     layer_size, len(top_rank), temp_confusion_matrix/5]
    # Ustawienie aktualnie najlepszego wyniku
    if best_score[0] < current_score[0]:
        best_score = deepcopy(current_score)

    list_score.append(current_score)
    return best_score


def main():
    # 1. Wyznaczenie rankingu cech
    files = glob.glob('sets/*.txt')
    X, Y = get_data(files)

    fvalue_selector = SelectKBest(f_classif)
    fvalue_selector.fit(X, Y)

    rank = fvalue_selector.scores_
    top_rank = []
    indexes = rank.argsort()[-16:][::-1]

    for index in indexes:
        top_rank.append(rank[index])

    # 2. Implementacja środowiska eksperymentowania

    # Format zapisu wyniku - [wynik, typ funkcji aktywacji, wartość momentum, rozmiar warstwy, macierz pomyłek]
    best_score = [0, '', 0, 0, 0, np.ndarray]
    # Tablica przechowująca wyniki wszystkich badań w formacie takim samym jak zmienna "best_score"
    list_score = []

    # Zdefiniowane trzech iczb neuronów w warstwie ukrytej
    layers = [100, 200, 300]
    # layers = [100]

    # Typy funkcji aktywacji
    activation_values = ['relu', 'logistic']
    # activation_values = ['relu']

    # Wartości momentum (domyślna wartość wynosi 0)
    momentum_values = [0, 0.9]
    # momentum_values = [0]

    # Całkowita liczba wszystkich badań
    total_examinations = len(momentum_values) * \
        len(activation_values) * len(layers) * 10

    # Indeks aktualnego badania
    current_examination = 1

    # Badania przeprowadzane przy użyciu wszystkich możliwych wcześniej zdefiniowanych kombinacji paramterów
    for layer in layers:
        for activation_value in activation_values:
            for momentum_value in momentum_values:
                for feature_number in range(1, 15):
                    to_train = indexes[:feature_number]

                    print(f'{current_examination}/{total_examinations} - start')

                    best_score = classification(X=X[:, to_train], Y=Y, activation_value=activation_value,
                                                momentum_value=momentum_value, top_rank=top_rank[
                                                    :feature_number - 1],
                                                fvalue_selector=fvalue_selector,
                                                layer_size=layer, best_score=best_score, list_score=list_score)

                    print(f'{current_examination}/{total_examinations} - end')
                    current_examination += 1

    dflist = pd.DataFrame(list_score)
    dflist.to_csv('wyniki.txt', encoding='utf-8', index=False)


if __name__ == "__main__":
    main()
