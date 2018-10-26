import numpy as np
import os
import pickle
from collections import Counter

import heapq

train_path = 'train.pkl'
test_path = 'test.pkl'

def readfile(path):
    '''
    reading content from path
    :param path:
    :return:
    '''
    f = open(path)
    content = pickle.load(f)
    return content

def Eucli(l1,l2):
    '''
    calculate the euclidean distance betwwen l1 and l2
    :param l1:
    :param l2:
    :return:
    '''
    euc_distance = np.sum(np.square(l1-l2))
    return euc_distance

def cosine(l1,l2):
    '''
    calculate the cosine distance betwwen l1 and l2
    :param l1:
    :param l2:
    :return: cosine
    '''
    l1_norm = np.sqrt(np.sum(np.square(l1)))
    l2_norm = np.sqrt(np.sum(np.square(l2)))
    prod = np.sum(l1 * l2)
    prod_norm = l1_norm * l2_norm
    cos_distance = float(prod) / prod_norm
    return cos_distance

def KNN():
    '''
    calculate the distance between the every test data and all training data
    choose appropriate K to get the nearest K point
    and get the most possible class
    :return:
    '''
    print('loading training and test data....')
    train = readfile(train_path)
    test = readfile(test_path)
    train_vec = np.array(train)[:, 1].astype(np.float64) #get [vector]
    test_vec = np.array(test)[:, 1].astype(np.float64)
    train_label = np.array(train)[:, 2] #get[label]
    test_label = np.array(test)[:, 2]

    # print(train_vec)
    # print(test_vec)
    print('data loading over...')

    K = 3
    index =0
    acc = 0
    for point in test_vec:
        sim = []
        print('the %d index test point vector',index)
        index +=1
        for pp in train_vec:
            sim.append(cosine(point,pp))
            # dis.append(Eucli(point,pp))
        max_num_index_list = map(sim.index, heapq.nlargest(K, sim)) # choose the most similar K points
        # min_num_index_list = map(dis.index, heapq.nsmallest(3, dis))
        dis = list(max_num_index_list)
        print(dis)
        label_list = train_label[dis]
        label_pred = Counter(label_list).most_common(1)
        label_truth = test_label[point]
        if label_pred==label_truth:

            acc+=1

    acc = float(acc) / len(test_vec)
    print('Accuracy on test: ', acc)

KNN()