# encoding=utf-8
import random
import numpy as np
import os
import re
import pickle
import json
import collections
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans, AffinityPropagation, MeanShift, SpectralClustering, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture

# corpus_path = 'Tweet/'

def readfile(r_path):
    # 避免编码错误ignore（python2和python3 不一致）
    with open(r_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    return content

def readfiles(root_path):
    '''
    用于读取文件夹下的所有文件
    :param root_path:
    :return:
    '''
    result = {}
    catelist = os.listdir(root_path)
    for mydir in catelist:
        class_path = root_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file in file_list:
            path = os.path.join(class_path, file)
            text = readfile(path)
            result[path] = text
    return result

def writefile(content,path):
    '''
    用于写入测试集和训练集
    :param content:
    :param path:
    :return:
    '''
    with open(path, 'w') as w:
        for line in content:
            w.write(line+'\n')

def writefile1(content,path):
    '''
    用于写入VSM过程中的一些文件
    :param content:
    :param path:
    :return:
    '''
    with open(path, 'wb') as wr:
        pickle.dump(content, wr)  # 把dic保存到pickle文件里

def train_split(root_path,train_path,test_path):
    '''
    train/test split(80% & 20%)
    :param root_path:
    :param train_path:
    :param test_path:
    :return:
    '''
    content = readfile(root_path).splitlines()
    train_set = []
    test_set  = []
    for line in content:
        rand = random.uniform(0, 1)
        if rand > 0.8:
            test_set.append(line)
        else:
            train_set.append(line)
    writefile(train_set, train_path)
    writefile(test_set, test_path)

def VSM(train_path,test_path,train_vec,test_vec):
    '''
    Vector Space Model
    :param train_path:
    :param test_path:
    :return:
    '''
    train = readfile(train_path).splitlines() #train[]
    test = readfile(test_path).splitlines()  # test[]

    dict = []
    for line in train:
        row = json.loads(line)
        row1 = row['text'].split(' ') # 通过json对文件格式进行处理
        dict.extend(row1) # extend not append !!!
    word_dict = list(set(dict))
    word_dict.sort()
    len_word_dict = len(word_dict)
    print('***********生成的词典大小是***************',len_word_dict)
    dict_path = 'Tweet/Dictionary.pkl'
    writefile1(word_dict, dict_path)

    train_tf = []
    test_tf = []
    for line in train:  # train_tf_vector space
        train_row = json.loads(line)
        train_row_text = train_row['text'].split(' ')
        train_list = collections.Counter(train_row_text)
        print(train_list)
        train_tf_vector = [0] * len_word_dict # initialize
        for i in range(len_word_dict):
            if word_dict[i] in train_list.keys():
                train_tf_vector[i] = train_list[word_dict[i]]
        train_tf_vector.append(train_row['cluster'])
        train_tf.append(train_tf_vector)
    for line in test: # train_tf_vector space
        test_row = json.loads(line)
        test_row_test = test_row['text'].split(' ')
        test_list = collections.Counter(test_row_test)
        print(test_list)
        test_tf_vector = [0] * len_word_dict # initialize
        for i in range(len_word_dict):
            if word_dict[i] in test_list.keys():
                test_tf_vector[i] = test_list[word_dict[i]]
        test_tf_vector.append(test_row['cluster'])
        test_tf.append(test_tf_vector)

    writefile1(train_tf, train_vec)
    writefile1(test_tf, test_vec)

def evaluate(algorithm_name, algorithm, y_train, y_test,X_train,X_test):
    """
    使用NMI指标评价聚类算法在训练集和测试集上的效果
    :param algorithm_name: str类型，表示聚类算法
    :param algorithm: sklearn聚类模型
    :param y_train: list或者一维numpy.ndarray类型，训练集的真实聚类标志
    :param y_test: list或者一维numpy.ndarray类型，测试集的真实聚类标志
    :return: None
    """
    y_train_pred = algorithm.predict(X_train)
    y_test_pred = algorithm.predict(X_test)
    print('this is : ', algorithm_name)
    print("NMI (train):", metrics.normalized_mutual_info_score(y_train, y_train_pred))
    print("NMI (test):",metrics.normalized_mutual_info_score(y_test, y_test_pred))

def evaluate2(algorithm_name, algorithm, X_train, y_train, X_test, y_test):
    """
    使用NMI指标评价聚类算法在训练集和测试集上的效果，聚类算法algorithm可能不含predict方法
    :param algorithm_name: str类型，表示聚类算法
    :param algorithm: sklearn聚类模型
    :param X_train: numpy.ndarray类型，训练集数据
    :param y_train: list或者一维numpy.ndarray类型，训练集的真实聚类标志
    :param X_test: numpy.ndarray类型，测试集数据
    :param y_test: list或者一维numpy.ndarray类型，测试集的真实聚类标志
    :return: None
    """
    y_train_pred = algorithm.fit_predict(X_train)
    X = np.row_stack((X_train, X_test))
    length_of_train = len(y_train)
    y_test_pred = algorithm.fit_predict(X)[length_of_train:]
    print('----------------------------------------------')
    print('Algorithm: ', algorithm_name)
    print("训练集上NMI值：%f" % metrics.normalized_mutual_info_score(y_train, y_train_pred))
    print("测试集上NMI值：%f" % metrics.normalized_mutual_info_score(y_test, y_test_pred))

def cluster(train_path,test_path):

    train = pickle.load(open(train_path,'rb')) # 读取pkl文件
    test = pickle.load(open(test_path,'rb'))

    train = np.array(train, dtype=np.int32)
    test = np.array(test, dtype=np.int32)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    cluster = len(set(y_train))
    print('train_set cluster nums:',cluster)

    kmeans = KMeans(n_clusters=cluster, random_state=0).fit(X_train)
    evaluate('K-Means', kmeans, y_train, y_test,X_train,X_test)

    ap = AffinityPropagation().fit(X_train)
    evaluate('AP', ap, y_train, y_test,X_train,X_test)

    ms = MeanShift(n_jobs=8).fit(X_train)
    evaluate('Mean Shift', ms, y_train, y_test,X_train,X_test)

    sc = SpectralClustering(n_clusters=cluster, assign_labels="discretize", random_state=0).fit(X_train)
    evaluate2('Spectral Clustering', sc, X_train, y_train, X_test, y_test)

    clustering = AgglomerativeClustering(n_clusters=cluster).fit(X_train)
    evaluate2('Ward Hierarchical Clustering', clustering, X_train, y_train, X_test, y_test)

    clustering = DBSCAN().fit(X_train)
    evaluate2('DBSCAN', clustering, X_train, y_train, X_test, y_test)

    clustering = GaussianMixture(n_components=cluster, random_state=0).fit(X_train)
    evaluate('Gaussian Mixture', clustering, y_train, y_test,X_train,X_test)

if __name__ == '__main__':
    path = 'Tweet/Tweets.txt'
    train_path = 'Tweet/train.txt'
    test_path = 'Tweet/test.txt'
    train_vec = 'Tweet/train.pkl'
    test_vec = 'Tweet/test.pkl'

    # train_split(path,train_path,test_path)
    # VSM(train_path,test_path,train_vec,test_vec)
    cluster(train_vec,test_vec)