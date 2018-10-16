# encoding=utf-8
import os
from textblob import TextBlob
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np
import random
import shutil

def split_data():
    '''
    split training set and testing set
    :return:
    '''
    corpus_path = '20news-18828/20news-18828/'
    trainPath = 'data/train/'
    testPath = 'data/test/'
    dict = {}
    catelist = os.listdir(corpus_path)  # get all subcategorys under this categroy
    sum = 0
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"  # get the path based subcategroy
        file_list = os.listdir(class_path)  # make a list about all sublist
        num = len(file_list)
        train_num = int(num * 0.8)
        print(mydir)
        print(train_num)
        sum += train_num
        train_list = random.sample(file_list, train_num)
        for doc in file_list:
            fullname = class_path + doc
            dict['name'] = doc
            dict['class'] = mydir
            src = os.path.join(class_path, doc)
            if doc in train_list:
                des = trainPath + mydir + '_' + doc
                shutil.copy(src, des)
            else:
                des = testPath + mydir + '_' + doc
                shutil.copy(src, des)
    print(sum)
    return dict

def readfile(path):
    fp = open(path, "r")
    content = fp.read()
    fp.close()
    return content

corpus_path = '20news-18828/20news-18828/'
temp_path = 'temp.txt'

def mergeData():
    '''
    merge all data into one txt
    '''
    catelist = os.listdir(corpus_path)  #get all subcategorys under this categroy
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"  #get the path based subcategroy
        file_list = os.listdir(class_path) #make a list about all sublist
        for file_path in file_list:
            fullname = class_path + file_path #path + XX.txt
            print("The text you read is:",fullname)  #corpus/train/pos/pos1.txt
            content = readfile(fullname)
            with open(temp_path, 'a+') as w:
                w.write(content)
    print('processing over......')

def process():
    '''
    data process including tokenization,stopwords filtering,stemming ,and get the dictionary
    :return:
    '''
    with open(temp_path,'r') as f:
        content = readfile(temp_path).decode('utf-8','ignore')
        # print(content)
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(content)
    # filtered_sentence = [w for w in word_tokens if not w in stop_words]

    filtered_sentence = set()
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.add(w)

    # vocab = set()
    # stemmer = PorterStemmer()  # Stemming
    # for word in filtered_sentence:
    #     new_word = stemmer.stem(word)
    #     vocab.add(new_word)

    print(len(filtered_sentence)) #279767
    return filtered_sentence


def doc2onehot_tf_matrix(vocab):
    '''
      transform document to onehot vector
    '''
    M = 18828  # 18828
    V = len(vocab)  # 279767
    onehot = np.zeros((M, V))
    docs =[]
    catelist = os.listdir(corpus_path)
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file_path in file_list:
            fullname = class_path + file_path  # path + XX.txt
            print("The text you read is:", fullname)  # corpus/train/pos/pos1.txt
            doc = readfile(fullname).decode('utf-8','ignore')
            docs.append(doc)
    for d, doc in enumerate(docs):
        for word in doc:
            if word in vocab:
                pos = vocab.index(word)
                onehot[d][pos] = 1
    return onehot


def random_copyfile(srcPath, dstPath, numfiles):
    name_list = list(os.path.join(srcPath, name) for name in os.listdir(srcPath))
    random_name_list = list(random.sample(name_list, numfiles))
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    for oldname in random_name_list:
        shutil.copyfile(oldname, oldname.replace(srcPath, dstPath))



words = process()
vocab = sorted(set(words), key=words.index)
doc2onehot_tf_matrix(vocab)

