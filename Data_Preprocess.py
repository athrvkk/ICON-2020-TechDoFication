import pandas as pd
import numpy as np
import csv
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.util import ngrams
from collections import Counter


class Data_Preprocess():
    
    # ----------------------------------------- Constructor -----------------------------------------
    
    def __init(self):
        self.punctuations = set(punctuation)
        
    
    # ------------------------------------------ Read Data ------------------------------------------
    
    def read_data(self, path):
        text = []
        label = []
        with open(path) as data_file:
            data = csv.reader(data_file, delimiter='\t', quoting=csv.QUOTE_NONE)
            next(data)
            for row in data:
                text.append(row[0])
                label.append(row[1])
        df = pd.DataFrame(list(zip(text, label)), columns=['text', 'label'])
        data_file.close()
        return df
    
     
    # ----------------------------------------- BOW Vectorizer -----------------------------------------
    
    def bow_vectorize(self, x_train, x_val):
            bow_vectorizer = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
            bow_vectorizer.fit(x_train)
            bow_x_train = bow_vectorizer.transform(x_train)
            bow_x_val = bow_vectorizer.transform(x_val)
            return bow_x_train, bow_x_val
        
    
    # ------------------------------------- bi/trigram TF-IDF Vectorizer -----------------------------------
    
    def tfidf_vectorize(self, x_train, x_val):
        tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        tfidf_vectorizer.fit(x_train)
        tfidf_x_train = tfidf_vectorizer.transform(x_train)
        tfidf_x_val = tfidf_vectorizer.transform(x_val)
        return tfidf_x_train, tfidf_x_val
    
    
    # ------------------------------------- unigram TF-IDF Vectorizer -----------------------------------
    
    def n_gram_tfidf_vectorize(self, x_train, x_val):
        n_gram_tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3))
        n_gram_tfidf_vectorizer.fit(x_train)
        n_gram_tfidf_x_train = n_gram_tfidf_vectorizer.transform(x_train)
        n_gram_tfidf_x_val = n_gram_tfidf_vectorizer.transform(x_val)
        return n_gram_tfidf_x_train, n_gram_tfidf_x_val
    
    
    # ---------------------------------- character-level TF-IDF Vectorizer --------------------------------
    
    def char_tfidf_vectorize(self, x_train, x_val):
        char_tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
        char_tfidf_vectorizer.fit(x_train)
        char_tfidf_x_train = char_tfidf_vectorizer.transform(x_train)
        char_tfidf_x_val = char_tfidf_vectorizer.transform(x_val)
        return char_tfidf_x_train, char_tfidf_x_val
    
     
    # ----------------------------------- Integer encoding labels ------------------------------------
        
    def label_encoder(self, y_train, y_test):
        le = LabelEncoder()
        le.fit(np.unique(y_train).tolist())
        y_train = le.transform(y_train)
        y_test = le.transform(y_test)
        return y_train, y_test
        
        
    # ----------------------------------------- Get unigrams -----------------------------------------
    
    def get_unigrams(self, corpus):  
        corpus = " ".join(x for x in corpus)
        corpus = [word for word in corpus.split() if word not in punctuation]
        unigrams = ngrams(corpus, 1)
        unigram_freq = Counter(unigrams)       
        return unigram_freq
    
    
     # ------------------------------------- Convert grams to dict ------------------------------------
        
    def counter_to_dict(self, counter):
        wordcloud_dict = {}
        gram = len(counter.most_common(1)[0][0])
        
        if gram == 1:
            for x in counter.most_common():
                wordcloud_dict[x[0][0]] = x[1]
        
        elif gram == 2:
            for x in counter.most_common():
                wordcloud_dict[x[0][0]+" "+x[0][1]] = round(x[1], 2)
                
        else:
            for x in counter.most_common():
                wordcloud_dict[x[0][0]+" "+x[0][1]+" "+x[0][2]] = round(x[1], 2)
        
        deleted = [key for key in wordcloud_dict if wordcloud_dict[key] < 2]
        for key in deleted: del(wordcloud_dict[key])
        
        return wordcloud_dict
