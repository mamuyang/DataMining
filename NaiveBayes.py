# encoding=utf-8
import numpy as np

import pickle

tf_path = 'tf_dict.pkl'
voc_path = 'word_dict_new.pkl'
train_index_path = 'data/train_index.pkl'
test_index_path = 'data/test_index.pkl'
train_tf_path = 'data/train_tf.pkl'
test_tf_path = 'data/test_tf.pkl'
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

def writefile(content,path):
    '''
    write content to path
    :param content:
    :param path:
    :return:
    '''
    with open(path, 'wb') as wr:
        pickle.dump(content, wr)  # 把dic保存到pickle文件里

def tf_calculate():
    '''
    calculate the tf for training set and test set respectively
    :return:
    '''
    tf_dict = readfile(tf_path) #读取所有词频信息，每个doc存储一个[]
    voc = readfile(voc_path) #读取data词典,[,,,...]
    train_index_list = readfile(train_index_path) #读取train_index[name,label]
    test_index_list = readfile(test_index_path) #读取test_index[name,label]

    print('training set start....')
    all_train_tf = []
    k1 = 0
    for num in train_index_list:
        name = num[0]
        label = num[1]
        # print('name:',name)
        # print('label:',label)
        train_tf_vec = [name]
        train_tf_vec.extend([0]*len(voc)) #not append!!!
        train_tf_vec.extend([label])
        for i in range(len(voc)):
            if voc[i] in tf_dict[name].keys():
                train_tf_vec[i+1] = tf_dict[name][voc[i]]
        all_train_tf.append(train_tf_vec)
        print('this is the % d training doc' % k1)
        k1 = k1 + 1
        # print(test_space_tf)
        # print(np.array(test_space_tf))
    print('tf calculation of training set finish.') #15056 train doc
    writefile(all_train_tf, train_tf_path)

    print('test set start....')
    all_test_tf = []
    k2 = 0
    for num in test_index_list:
        name = num[0]
        label = num[1]
        # print('name:',name)
        # print('label:',label)
        test_tf_vec = [name]
        test_tf_vec.extend([0] * len(voc))
        test_tf_vec.extend(label)
        for i in range(len(voc)):
            if voc[i] in tf_dict[name].keys():
                test_tf_vec[i + 1] = tf_dict[name][voc[i]]
        all_test_tf.append(test_tf_vec)
        print('this is the % d test doc' % k2)
        k2 = k2 + 1
    print('tf calculation of test set finish.') # 3772 test doc
    writefile(all_test_tf, test_tf_path)

    return

def prior_compute(tf,voc_size):
    '''
    计算先验 P(term|C)和P(C)
    :param tf: 分别记录train和test的tf文件
    :param voc_size: 词典的大小
    :return:
    '''
    #(class label ranges from 0 to 19)
    class_list = [0]*20
    doc_of_class = []
    for i in range(20):
        print('this is the % d class:' % i)
        tmp_doc = np.where(tf[:,-1] == i)[0] #拿到同一类的所有doc
        # print(tmp_doc)
        class_list[i] = len(tmp_doc) #每一类的num_of_doc
        tmp_doc = np.sum(tf[tmp_doc, :-1], axis=0)
        doc_of_class.append(tmp_doc.tolist()) #每一类下面的term tf

    class_list = np.array(class_list)
    print(class_list) #[797 795 784 778 795 799 788 784 728 789 752 792 639 777 792 785 768 792 502 620]
    doc_of_class = np.array(doc_of_class)
    print(doc_of_class.shape) #[20,24227]
    print('sum of doc:',sum(class_list)) # 15056/3772
    # 计算 P(C)
    p_class = class_list/float(sum(class_list)) #float!!!!! otherwise all values are zero.
    print('each p of class:',p_class)
    # 计算 P(term|C),采用多项式平滑

    tmp_sum = doc_of_class.sum(axis=1) + voc_size #每一类的所有tf之和加上voc_size(平滑)
    print(tmp_sum)
    p_term =[]
    for i in range(doc_of_class.shape[0]):
        m = list((doc_of_class[i]+1)/float(tmp_sum[i]))
        p_term.append(m)
    p_term = np.array(p_term) #每一类的 n+1/ N+voc_size
    print('p_term:',p_term)
    # 拿到 P(term|C),[20 * (voc_size + 1)]
    p_term_c = np.column_stack((p_term, p_class.reshape((-1, 1))))
    # 拿到 log(P(term|C)),[20 * (voc_size + 1)]
    p_term_c_log = np.log10(p_term_c)

    p_path = 'data/p.pkl'
    p_log_path = 'data/p_log.pkl'
    writefile(p_term_c, p_path)
    writefile(p_term_c_log, p_log_path)
    return p_term_c_log

def train_NB():
    '''
    对训练集计算朴素贝叶斯分类器
    :return:
    '''
    voc = readfile(voc_path)
    voc_size = len(voc)  # 24227
    train_tf = readfile(train_tf_path)
    train_tf = np.array(train_tf)[:, 1:].astype(np.int32)
    print(train_tf.shape)  # [15056,24228]
    print('test_tf read over.')
    p_term_c_log = prior_compute(train_tf, voc_size)
    print('compute log of p_term over.')

    print('likelyhood function calculation...')
    # 对每一个文档，计算他的极大似然那函数，然后判断属于哪一类
    count = 0
    for i in range(train_tf.shape[0]):
        f_likelihood = train_tf[i, :-1] * p_term_c_log[:, :-1]
        if np.argmax(np.sum(f_likelihood, axis=1) + p_term_c_log[:, -1]) == train_tf[i, -1]:
            count = count + 1
    print('predict True:', count)
    acc = float(count) / 15056
    print('accuarcy is:', acc)

def test_NB():
    '''
    对测试集计算朴素贝叶斯分类器
    :return:
    '''
    voc = readfile(voc_path)
    voc_size = len(voc)  # 24227
    test_tf = readfile(test_tf_path)
    test_tf = np.array(test_tf)[:, 1:].astype(np.int32)
    print(test_tf.shape)  # [3772,24228]
    print('test_tf read over.')
    p_term_c_log = prior_compute(test_tf, voc_size)
    print('compute log of p_term over.')

    print('likelyhood function calculation...')
    # 对每一个文档，计算他的极大似然那函数，然后判断属于哪一类
    count = 0
    for i in range(test_tf.shape[0]):
        f_likelihood = test_tf[i, :-1] * p_term_c_log[:, :-1]
        if np.argmax(np.sum(f_likelihood, axis=1) + p_term_c_log[:, -1]) == test_tf[i, -1]:
            count = count + 1
    print('predict True:', count)
    acc = float(count) / 3772
    print('accuarcy is:', acc)

    train_tf = readfile(train_tf_path)
    train_tf = np.array(train_tf)[:, 1:].astype(np.int32)
    print(train_tf.shape)  # []
    print('train_tf read over.')
    p_term_c_log = prior_compute(train_tf, voc_size)
    print('compute log of p_term over.')
    p_term_c_log = prior_compute(test_tf, voc_size)
    print('compute log of p_term over.')

    print('likelyhood function calculation...')
    # 对每一个文档，计算他的极大似然那函数，然后判断属于哪一类
    count = 0
    for i in range(test_tf.shape[0]):
        f_likelihood = test_tf[i, :-1] * p_term_c_log[:, :-1]
        if np.argmax(np.sum(f_likelihood, axis=1) + p_term_c_log[:, -1]) == test_tf[i, -1]:
            count = count + 1
    print('predict True:', count)
    acc = float(count) / 3772
    print('accuarcy is:', acc)
    return acc
if __name__ == '__main__':

    # tf_calculate()
    train_NB()
    # test_NB()



