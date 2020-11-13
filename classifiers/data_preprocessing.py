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
from collections import Counter
import unicodedata
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel



# ------------------------------------------ Read Data ------------------------------------------

def read_data(path):
    return pd.read_csv(path)



# ------------------------------------------ Get Stopwords ------------------------------------------
    
def get_stopwords(path):
    file = open(path, "r")
    stopword_list = []
    for row in file:
        row = row.replace("\n", "")
        stopword_list.append(row)
    return stopword_list
        
    
    
# ------------------------------------------ Clean Data ------------------------------------------ 

def clean_text(text):
    #Removing unprintable characters
    text = ''.join(x for x in text if x.isprintable())

    # Cleaning the urls
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Cleaning the html elements
    text = re.sub(r'<.*?>', '', text)

    # Removing the punctuations
    text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', '', text)
    
    text = " ".join(word.lower() for word in text.split())
    return text



# ------------------------------------------ Preprocess Data ------------------------------------------

def preprocess_data(stopword_list, text: str) -> str:
    
    def expand_concatenations(word):            
        if not re.match('[a-zA-Z]+', word) or re.match('\d+',word):
            for i in range(len(word)):
                if not('DEVANAGARI ' in unicodedata.name(word[i])):
                    word = word[:i] if( len(word[i:]) < 2 and not word[i:].isnumeric()) else word[:i] + " " + word[i:]
                    break
        else:
            for i in range(len(word)):
                if ('DEVANAGARI ' in unicodedata.name(word[i])):
                    word = word[i:] if( len(word[:i]) < 2 and not word[:i].isnumeric() ) else word[:i] + " " + word[i:]
                    break
        return(word)
    
    
    try:
        if not(isinstance(text, str)): text = str(text)

        #Removing unprintable characters
        text = ''.join(x for x in text if x.isprintable())

        # Cleaning the urls
        text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Cleaning the html elements
        text = re.sub(r'<.*?>', '', text)

        # Removing the punctuations
        text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', '', text)


        # Removing stop words
        text = ' '.join([word for word in text.split() if word not in stopword_list])

        # Expanding noisy concatenations (Eg: algorithmआणि  -> algorithm आणि ) 
        text = ' '.join([expand_concatenations(word) for word in text.split()])

        preprocessed_text = ""

        for word in text.split(): 
            if (re.match('\d+', word)):
                if(word.isnumeric()):
                    preprocessed_text = preprocessed_text + '#N' + " "
            else:
                if(re.match('[a-zA-Z]+', word)):
                    if not len(word) < 2:
                        preprocessed_text = preprocessed_text + word.lower() + " "
                else:
                    preprocessed_text = preprocessed_text + word + " "

        return preprocessed_text.strip()

    except ValueError as ve:
        print('Error processing:\t',text)
        return ''        
    
    
    
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



# ---------------------------------- character-level Count Vectorizer --------------------------------

def char_bow_vectorize(x_train, x_val, min_df):
        char_bow_vectorizer = CountVectorizer(analyzer='char', ngram_range=(2,3))
        char_bow_vectorizer.fit(x_train)
        char_bow_x_train = char_bow_vectorizer.transform(x_train)
        char_bow_x_val = char_bow_vectorizer.transform(x_val)
        return char_bow_vectorizer, char_bow_x_train, char_bow_x_val
    
    
    
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



# ----------------------------------------- tokenizing -----------------------------------------

def tokenize_text(x_train, x_val):
    tokenizer = Tokenizer(oov_token='[OOV]')
    tokenizer.fit_on_texts(x_train)
    x_train_tokenzied = tokenizer.texts_to_sequences(x_train)
    x_val_tokenzied = tokenizer.texts_to_sequences(x_val)
    return tokenizer, x_train_tokenzied, x_val_tokenzied
    
    
    
# ----------------------------------------- Pading and Truncating -----------------------------------------
    
def pad_text(x_train_tokenzied, x_val_tokenzied, pad_len, padding_type='post', truncating_type='post'):    
    x_train_padded = np.asarray(pad_sequences(x_train_tokenzied, 
                                              padding=padding_type, 
                                              truncating=truncating_type, 
                                              maxlen=pad_len))
    x_val_padded = np.asarray(pad_sequences(x_val_tokenzied, 
                                            padding=padding_type, 
                                            truncating=truncating_type, 
                                            maxlen=pad_len))
    return x_train_padded, x_val_padded



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
    embedding_file = open(embedding_path, 'r',encoding = 'utf8')
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

    
# ----------------------------------------- Prepare Data for LDA input -----------------------------------------

def prepare_LDA_input(corpus, LDA_model):
    # Prepare input to LDA model
    corpus = [clean_text(text).split() for text in corpus]
    dict_corpus = Dictionary(corpus)
    dict_corpus.filter_extremes(no_below=5, no_above=0.3, keep_n=None)
    bow_corpus = [dict_corpus.doc2bow(c) for c in corpus]
    
    # Get topic-doc vector
    LDA_input = []
    for doc in bow_corpus:
        LDA_input.append(LDA_model.get_document_topics(doc))
    
    # Add missing probabilities
    for doc in LDA_input:
        index = []
        true_index = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        for i in range(len(doc)):
            index.append(doc[i][0])
        new_index = true_index- set(index)
        for j in new_index:
            doc.extend([(j, 0.0)])
        doc.sort() 
        
    # Create input matrix
    LDA_doc = []
    for doc in LDA_input:
        LDA_doc.append(np.asarray([doc[0][1], doc[1][1], doc[2][1], doc[3][1],
                                   doc[4][1], doc[5][1], doc[6][1], doc[7][1],
                                   doc[8][1], doc[9][1], doc[10][1], doc[11][1]], dtype='float32'))
    LDA_doc = np.array(LDA_doc)
    return LDA_doc



# ----------------------------------------- Get unigrams -----------------------------------------

def get_unigrams(corpus):
    unigram_freq = {}
    corpus = " ".join(x for x in corpus)
    corpus = [word for word in corpus.split() if word not in punctuation]
    unigrams = ngrams(corpus, 1)
    for x in Counter(unigrams).most_common():
        unigram_freq[x[0][0]] = x[1]
    return unigram_freq
