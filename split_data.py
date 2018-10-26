# encoding=utf-8
import os
import numpy as np
import random
import shutil
import pickle


corpus_path = '20news-18828/20news-18828/'

def writefile(content,path):
    '''
    write content to the path
    :param content:
    :param path:
    :return:
    '''
    with open(path, 'wb') as wr:
        pickle.dump(content, wr)

def split_data():
    '''
    load the vector space represenetation
    randomly split the data into training set(80%) and testing set(20%)
    and save data into train[name,vector],test[name,vector]
    :return:
    '''
    trainPath = 'data/train/'
    testPath = 'data/test/'

    train_index = []  #[name,label]
    test_index = []   #[name,label]
    class_index = 0   # record class index (0-19)

    sum = 0
    catelist = os.listdir(corpus_path)  # get all subcategorys under this categroy
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        file_list = os.listdir(class_path)
        num = len(file_list)
        train_num = int(num * 0.8)  # random sampling (80%)
        print('mydir:',mydir)
        print('choose the train_num:',train_num) # random sampling number of every class
        sum += train_num
        train_list = random.sample(file_list, train_num)
        for doc in file_list:
            fullname = class_path + doc
            list = [fullname, class_index]
            src = os.path.join(class_path, doc)
            if doc in train_list:
                des = trainPath + mydir + '_' + doc
                shutil.copy(src, des)  # copy training files to the destination path
                train_index.append(list)
            else:
                des = testPath + mydir + '_' + doc
                shutil.copy(src, des) # copy test files to the destination path
                test_index.append(list)
    class_index += 1
    writefile(train_index, 'data/train_index.pkl')
    writefile(test_index, 'data/test_index.pkl')
    print('Total training set size:',sum)  #15056

    print('loading doc vector space representation... ')
    vsm = 'vector_space.pkl'
    f = open(vsm)
    vector = pickle.load(f)

    print('saving training set and testing set ...')
    train_data = []  # [name,vactor,label]
    test_data = []   #[name,vector,label]

    train_key = np.array(train_index)[:, 0].tolist() #get all vector name
    test_key = np.array(test_index)[:, 0].tolist()

    for content in vector:
        name = content[0]
        if name in train_key:
            index = train_key.index(name)
            content.append(train_index[index][-1])
            train_data.append(content)
        else:
            index = test_key.index(name)
            content.append(test_index[index][-1])
            test_data.append(content)

    writefile(train_data,'train.pkl')
    writefile(test_data,'test.pkl')
    print('data split over......')


split_data()