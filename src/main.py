import sys
import random
import csv
import numpy as np
from numpy.core.arrayprint import printoptions

from cluster import Cluster

def readCSV(csv_file_path:str):
    _data = np.loadtxt(csv_file_path, delimiter = ',')
    _todohuken:list = []
    with open('data/todohuken.csv') as f:
        _reader = csv.reader(f)
        _todohuken = [row for row in _reader][0]

    return _data, _todohuken

def k_means(data:np.ndarray, k:int):
    if np.size(data, axis = 0) < k:
        print('Error: クラスタ数がサンプル数よりも多いです')
        return
    elif k < 1:
        print('Error: クラスタ数が不正です')
    
    # 初期化処理
    _clusters:list = []
    _euclidean_distance_sum = 0
    for representative_vector in data[tuple([random.sample(range(len(data)), k = k)])]:
        _clusters.append(Cluster(representative_vector))
    
    # 学習
    while not all([cluster.isLearningComplete for cluster in _clusters]):
        for cluster in _clusters:
            cluster.initElements()
        _clusters = divideData(data, _clusters)
        for cluster in _clusters:
            cluster.recalRepresentativeVector()
    
    for cluster in _clusters:
        for element in cluster.elements:
            _euclidean_distance_sum += cluster.euclideanDistance(element)

    return _clusters, _euclidean_distance_sum
    
def divideData(data:np.ndarray, clusters:list[Cluster]):
    _clusters = clusters
    for i, row in enumerate(data):
        assignment_index = 0
        for j, cluster in enumerate(_clusters):
            if cluster.euclideanDistance(row) < _clusters[assignment_index].euclideanDistance(row):
                assignment_index = j
        _clusters[assignment_index].setElement(i, row)

    return _clusters

def printLearnResult(_clusters:list[Cluster], _todohuken:np.ndarray, k:int):
    print('クラスタ数: {0}'.format(k))
    for i, cluster in enumerate(_clusters):
        print('--- クラスタ{0} ---'.format(i + 1))
        print('代表ベクトル: ', cluster.representative_vector)

        _divided_todohuken = ''
        for i, todohuken_index in enumerate(cluster.elements_index):
            if i == 0:
                _divided_todohuken += '{0}'.format(_todohuken[todohuken_index])
            else:
                _divided_todohuken += ', {0}'.format(_todohuken[todohuken_index])
        print('分類都道府県: ', _divided_todohuken)
        print('\n')

def main(csv_file_path:str, k:int):
    _data, _todohuken = readCSV(csv_file_path)
    _clusters = []
    _euclidean_distance_sum = 0
    for i in range(50):
        _index_clusters, _index_euclidean_distance_sum = k_means(_data, k)
        if i == 0 or _index_euclidean_distance_sum < _euclidean_distance_sum:
            _clusters = _index_clusters
            _euclidean_distance_sum = _index_euclidean_distance_sum
    printLearnResult(_clusters, _todohuken, k)

if __name__ == '__main__':
    # _csv_file_path = sys.stdin[1]
    # _k = sys.stdin[2]
    _csv_file_path = 'data/sangyohi.csv'
    _k = 6

    main(_csv_file_path, _k)