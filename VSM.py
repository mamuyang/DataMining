# encoding=utf-8
import os
import nltk as nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle
import collections
import math

corpus_path = '20news-18828/20news-18828/'
temp_path = 'temp.txt'


def readfile(path):
    '''
    reading content from path
    :param path:
    :return:
    '''
    with open(path, 'r') as f:
        content = f.read().decode('utf-8','ignore')
    return content

def readfiles():
    '''
    reading file based root-dict
    :return:
    '''
    result = {}
    catelist = os.listdir(corpus_path)
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        file_list = os.listdir(class_path)
        for file in file_list:
            path = os.path.join(class_path, file)
            text = readfile(path)
            result[path] = text
    return result

def writefile(content,path):
    '''
    write content to path
    :param content:
    :param path:
    :return:
    '''
    with open(path, 'wb') as wr:
        pickle.dump(content, wr)  # 把dic保存到pickle文件里

# def mergeData():
#     '''
#     merge all data into one txt
#     '''
#     catelist = os.listdir(corpus_path)  #get all subcategorys under this categroy
#     for mydir in catelist:
#         class_path = corpus_path + mydir + "/"  #get the path based subcategroy
#         file_list = os.listdir(class_path) #make a list about all sublist
#         for file_path in file_list:
#             fullname = class_path + file_path #path + XX.txt
#             print("The text you read is:",fullname)  #corpus/train/pos/pos1.txt
#             content = readfile(fullname)
#             with open(temp_path, 'a+') as w:
#                 w.write(content)
#     print('processing over......')

def process():
    '''
    data-process including tokenization,stopwords filtering,stemming(version unmatched???)
    get the dictionary and vector representation
    :return:
    '''

    text_dict = readfiles()
    writefile(text_dict, 'text_dict.pkl')
    word_tokens =[]
    tf_dict = {}
    for doc in text_dict:
        content = text_dict[doc]
        tokens = word_tokenize(content)
        word_tokens.extend(tokens) # not append!!!!list convert to set cannot hash!!!
        tf_dict[doc] = collections.Counter(tokens)

    writefile(tf_dict,'tf_dict.pkl')
    print(len(word_tokens))

    word_dict = set(word_tokens)
    print(word_dict) #
    word_dict1 =set()
    stop_words = set(stopwords.words('english'))
    for w in word_dict:
        if w not in stop_words:
            word_dict1.add(w)
    word_dict = word_dict1
    print('word_dict:',len(word_dict)) #279765

    word_dict = list(word_dict)
    word_dict.sort()

    len_word_dict = len(word_dict) #279765
    num_of_doc = len(tf_dict) #18828
    print('num of doc:',num_of_doc)
    print('-------------df caculate-----------')
    df = [0] * len_word_dict
    for key in tf_dict:
        for i in range(len_word_dict):
            if word_dict[i] in tf_dict[key]:
                df[i] = df[i] + 1
    idf = [math.log(num_of_doc / t) for t in df]

    thresh = [1 if int(t) >= 50 else 0 for t in df]
    # setting threshhold to filter df with lower num

    word_dict_new = []
    df_new = []
    idf_new = []
    for i in range(len(df)):
        if thresh[i] == 1:
            word_dict_new.append(word_dict[i])
            df_new.append(df[i])
            idf_new.append(idf[i])
    # writefile(word_dict_new, 'word_dict_new.pkl')
    # writefile(df_new, 'df_new.pkl')
    # writefile(idf_new, 'idf_new.pkl')

    len_dict_new = len(word_dict_new) #6343
    print('The size of new dict is:',len_dict_new)
    vector_space = []
    print('start building vector space model...')
    for key in tf_dict:
        vector = [key]
        vector.extend([0] * len_dict_new)
        for i in range(len_dict_new):
            if word_dict_new[i] in tf_dict[key]:
                vector[i + 1] = (1 + math.log(tf_dict[key][word_dict_new[i]])) * idf_new[i]  # tf * idf
        vector_space.append(vector)
    writefile(vector_space, 'vector_space.pkl')
    print('vector space model build over......')
    return

# def doc2onehot_tf_matrix(vocab):
#     '''
#       transform document to onehot vector
#     '''
#     M = 18828  # 18828
#     V = len(vocab)  # 279767
#     onehot = np.zeros((M, V))
#     docs =[]
#     dict = {}
#     catelist = os.listdir(corpus_path)
#     for mydir in catelist:
#         class_path = corpus_path + mydir + "/"
#         file_list = os.listdir(class_path)
#         for file_path in file_list:
#             fullname = class_path + file_path  # path + XX.txt
#             print("The text you read is:", fullname)  # corpus/train/pos/pos1.txt
#             dict['name'] = fullname
#             dict['class'] = mydir
#             doc = readfile(fullname).decode('utf-8','ignore')
#             dict['content'] = doc
#             docs.append(doc)
#     for d, doc in enumerate(docs):
#         for word in doc:
#             if word in vocab:
#                 pos = vocab.index(word)
#                 onehot[d][pos] = 1
#     print('matrix build over.....')
#     with open('data/dict.pickle','wb') as wr:
#         pickle.dump(dict,wr)
#     with open('data/doc2vec.pickle', 'wb') as wr:
#         pickle.dump(onehot, wr)  # 把dic保存到pickle文件里
#
#     return onehot

if __name__ == '__main__':
    process()
