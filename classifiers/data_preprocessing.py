import pandas as pd
import numpy as np
import re
import csv
import regex
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from nltk.util import ngrams
from nltk.stem import WordNetLemmatizer
from collections import Counter
import unicodedata




# ------------------------------------------ Read Data ------------------------------------------

def read_data(path):
    return pd.read_csv(path)


 # --------------------------------------- Expand concatenations --------------------------------------

    
    
    
# ------------------------------------------ Preprocess Data ------------------------------------------

def preprocess_data(text, stopword_list):
    
    def expand_concatenations(word):
        if re.match('[a-zA-Z]+', word):
            for i in range(len(word)):
                if('DEVANAGARI ' in unicodedata.name(word[i])):
                    word = word[i:] if( len(word[:i]) < 2 ) else word[:i] + " " + word[i:]
                    break
        return(word)
    
    # Cleaning the urls
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))

    # Cleaning the html elements
    text = re.sub(r'<.*?>', '', text)

    # Removing the punctuations
    text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', ' ', text)

    # Removing stop words
    text = [word for word in text.split() if word not in stopword_list]

    # Expanding noisy concatenations (Eg: algorithmआणि  -> algorithm आणि ) 
    text = [expand_concatenations(word) for word in text]

    preprocessed_text = ""
    for word in text: 
        if (re.match('\d+', word)):
            if(word.isnumeric()):
                preprocessed_text = preprocessed_text + '<Numeric>' + " "

        else:
            if(re.match('[a-zA-Z]+', word)):
                word = word.lower()
                preprocessed_text = preprocessed_text + word + " "

            else:
                preprocessed_text = preprocessed_text + word + " "

    return preprocessed_text.strip() 
    
    
# ----------------------------------------- BOW Vectorizer -----------------------------------------

def custom_analyzer(text):
    # extract words of at least 1 letters
    words = regex.findall(r'\w{1,}', text)
    for w in words:
        yield w

        
def bow_vectorize(x_train, x_val, min_df):
        bow_vectorizer = CountVectorizer(analyzer=custom_analyzer, min_df=min_df)
        bow_vectorizer.fit(x_train)
        bow_x_train = bow_vectorizer.transform(x_train)
        bow_x_val = bow_vectorizer.transform(x_val)
        return bow_vectorizer, bow_x_train, bow_x_val


# ------------------------------------- bi/trigram TF-IDF Vectorizer -----------------------------------

def tfidf_vectorize(x_train, x_val, min_df):
    tfidf_vectorizer = TfidfVectorizer(analyzer=custom_analyzer, min_df=min_df)
    tfidf_vectorizer.fit(x_train)
    tfidf_x_train = tfidf_vectorizer.transform(x_train)
    tfidf_x_val = tfidf_vectorizer.transform(x_val)
    return tfidf_vectorizer, tfidf_x_train, tfidf_x_val


# ------------------------------------- unigram TF-IDF Vectorizer -----------------------------------

def n_gram_tfidf_vectorize(x_train, x_val, min_df):
    n_gram_tfidf_vectorizer = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), min_df=min_df)
    n_gram_tfidf_vectorizer.fit(x_train)
    n_gram_tfidf_x_train = n_gram_tfidf_vectorizer.transform(x_train)
    n_gram_tfidf_x_val = n_gram_tfidf_vectorizer.transform(x_val)
    return n_gram_tfidf_vectorizer, n_gram_tfidf_x_train, n_gram_tfidf_x_val


# ---------------------------------- character-level TF-IDF Vectorizer --------------------------------

def char_tfidf_vectorize(x_train, x_val):
    char_tfidf_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,3))
    char_tfidf_vectorizer.fit(x_train)
    char_tfidf_x_train = char_tfidf_vectorizer.transform(x_train)
    char_tfidf_x_val = char_tfidf_vectorizer.transform(x_val)
    return char_tfidf_vectorizer, char_tfidf_x_train, char_tfidf_x_val


# ----------------------------------- Integer encoding labels ------------------------------------

def label_encoder(y_train, y_test):
    le = LabelEncoder()
    le.fit(np.unique(y_train).tolist())
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    return y_train, y_test


# ----------------------------------------- Get Word Embeddings -----------------------------------------

def get_embedding_matrix(embedding_path, vocab, embedding_dim):
    cnt = 0
    vocab_words = set(vocab.keys())
    embedding_matrix = np.zeros((len(vocab)+1, embedding_dim))
    embedding_file = open(embedding_path, 'r')
    for row in embedding_file:
        row = row.split()
        word = row[0].strip()
        if word in vocab_words:
            wv = np.asarray(row[1:], dtype='float32')
            if len(wv) == embedding_dim:
                embedding_matrix[vocab[word]] = wv
                cnt = cnt + 1
    print(cnt)
    embedding_file.close()
    return embedding_matrix


# ----------------------------------------- Get Sentence Embeddings -----------------------------------------


def get_sentence_embedding(embedding_matrix, corpus, option='bow'):
    all_sentence_embeddings = []
    if option == 'bow':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            if row.data.shape[0] != 0:
                sentence_embedding = sentence_embedding/row.data.shape[0]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
        
    elif option == 'tfidf':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
    
    else:
        print("Invalid option")
        return text
    

# ----------------------------------------- tokenizer and pad for neural networks -----------------------------------------
    
def tokenizer_and_pad_training(x_train, x_val, pad_len, padding_type='post', truncating_type='post'):
    tokenizer = Tokenizer(oov_token='[OOV]')
    tokenizer.fit_on_texts(x_train)
    x_train_padded = tokenizer.texts_to_sequences(x_train)
    x_val_padded = tokenizer.texts_to_sequences(x_val)
    
    x_train_padded = np.asarray(pad_sequences(x_train_padded, 
                                              padding=padding_type, 
                                              truncating=truncating_type, 
                                              maxlen=pad_len))
    x_val_padded = np.asarray(pad_sequences(x_val_padded, 
                                            padding=padding_type, 
                                            truncating=truncating_type, 
                                            maxlen=pad_len))
    return tokenizer, x_train_padded, x_val_padded


# ----------------------------------------- Get unigrams -----------------------------------------

def get_unigrams(corpus):  
    corpus = " ".join(x for x in corpus)
    corpus = [word for word in corpus.split() if word not in punctuation]
    unigrams = ngrams(corpus, 1)
    unigram_freq = Counter(unigrams)       
    return unigram_freq


 # ------------------------------------- Convert grams to dict ------------------------------------

def counter_to_dict(counter):
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
