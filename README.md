# Data_Mining

############ Homework 1 ##############

This is the first homework.There are 2 tasks. One is to handle the text dataset,and get every single doc's vector representation.Another is to build KNN classifier based on this dataset.
20news-18828.tar is the dataset.
The 20 Newsgroups dataset is a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups. 

VSM.py is the data-process including tokenization,stopwords filtering,stemming(version unmatched???) , get the dictionary and vector representation. After the first token ,we get the size of word_dict is 279765, and filter the words that df < 50, the size of word_dict changes to 6343. Then get the vector space representation of every doc. And save the VSM into vector_space.pkl.(the file is too large to upload.) ( Several experiments demonstrate that df threshhold should not be too low.Otherwise,it will influence the accuracy of KNN.)

Split_data.py is loading the vector space represenetation,and randomly spliting the data into training set(80%) and testing set(20%), then saving data into train.pkl, test.pkl(the files are too large to upload.)

KNN.py use the vector space representation to calculate the distance between the every test data and all training data, and choose the appropriate K to get the nearest K point, and get the most possible class.


############ Homework 2 #################

This is the second homework. The task is implementing a naive bayes classifier, and evaluate the performance on the training set and test set respectively. 
code : NaiveBayes.py

###########  Homework3  ##################
This is the third homework. The task is processing the given tweets dataset to implementing some cluster algorithms by using the sklearn.

****The data format:
{{"text": "brain fluid buildup delay giffords rehab", "cluster": 37},
{"text": "trailer talk week movie rite mechanic week opportunity", "cluster": 14},
....}

****Algorithm:
////K-Means
////AP
////Mean-shift
////Spectral Cluster
////Ward hierarchial cluster
////DBSCAN
////GM(Memory Error???)

code:cluster_sklearn.py

