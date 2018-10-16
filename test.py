#encoding : utf-8
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import random
import shutil
# stemmer = PorterStemmer()
# print(stemmer.stem('working'))
# print(stemmer.stem('worked'))

# nltk.download("stopwords")

# with open('test.txt','w') as w:
#     w.write('ghlkjivbh')
# with open('test.txt','r') as r:
#     print(r.read())

# learning about textblob

# path = 'test.txt'
# with open(path,'r') as f:
#     content = f.read()
# wiki = TextBlob(content)
# print(wiki.words)

# wiki = TextBlob("Python is a high-level, general-purpose programming language.")
# print(wiki.noun_phrases)
# zen = TextBlob("Beautiful is better than ugly. "
#                "Explicit is better than implicit. "
#                "spaces information went on.")
# senlist = zen.sentences
# print('Senteces:',senlist)
# wordlist = zen.words
# print('Tokenization:',wordlist)
# IScount =0
# for sen in senlist:
#     print(sen.words.singularize())
#     print(sen.ngrams(n=3))
#     IScount += sen.words.count('is')
# print(IScount)
# def readfile(path):
#     fp = open(path, "r")
#     content = fp.read()
#     fp.close()
#     return content
#
# corpus_path = '20news-18828/20news-18828/'
# temp_path = 'temp1.txt'
# test_path = 'test.txt'


# catelist = os.listdir(corpus_path)  #get all subcategorys under this categroy
# for mydir in catelist:
#     class_path = corpus_path + mydir + "/"  #get the path based subcategroy
#     file_list = os.listdir(class_path) #make a list about all sublist
#     for file_path in file_list:
#         fullname = class_path + file_path #path + XX.txt
#         print("The text you read is:",fullname)  #corpus/train/pos/pos1.txt
#         content = readfile(fullname)
#         with open(temp_path, 'a+') as w:
#             w.write(content)
# print('processing over......')
#
# with open(temp_path,'r') as f:
#     content = readfile(temp_path)
#     print(content)

# path = 'test.txt'
# with open(path,'r') as f:
#     content = f.read().decode('utf-8','ignore')
# wiki = TextBlob(content)
# print(wiki.words)

corpus_path = '20news-18828/20news-18828/'
trainPath = 'data/train/'
testPath = 'data/test/'
dict ={}
catelist = os.listdir(corpus_path)  # get all subcategorys under this categroy
sum =0
for mydir in catelist:
    class_path = corpus_path + mydir + "/"  # get the path based subcategroy
    file_list = os.listdir(class_path)  # make a list about all sublist
    num = len(file_list)
    train_num = int(num * 0.8)
    print(mydir)
    print(train_num)
    sum +=train_num
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
            shutil.copy(src,des)
print(sum)


