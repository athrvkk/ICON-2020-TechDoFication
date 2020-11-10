#!/usr/bin/env python
# coding: utf-8

# In[31]:


import fasttext.util
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from gensim.models.fasttext import FastText 
from gensim.models import LsiModel
import random
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from time import process_time
import errno,pickle
import numpy as np
import codecs


# In[26]:


class TextToTensor:

    # --------------------------------------- Constructor --------------------------------------- 
    
    def __init__(self, tokenizer, max_len):
    
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def string_to_tensor(self, string_list: list) -> list:
        """
        A method to convert a string list to a tensor for a deep learning model
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list


# In[118]:


class Embeddings:

    """
    A class to read the word embedding file and to create the word embedding matrix
    """

#     def __init__(self, path, vector_dimension):
#         self.path = path 
#         self.vector_dimension = vector_dimension
    
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    
    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    
    def create_embedding_matrix(self, tokenizer, max_features):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        
        return embedding_matrix
    
    def save_embeddings(self,model,filepath):
        
        if (".vec" in filepath or ".txt" in filepath):
            file = open(filepath, "w",encoding= 'utf-8')
            words = model.keys()
            cnt = 0
            for w in words:
                v = model[w]
                vstr = ""
                for value in v:
                    vstr += " " + str(value)
                try:
                    row = w + vstr + "\n"
                    file.write(row)
                    cnt += 1
                except Exception as e:
                    print('Exception: ',e)
#                     if e.errno == errno.EPIPE:
#                         pass
            print('Words processed: ',cnt)

        
        elif ".plk" in filepath:
            with open(filepath,'wb') as file:
                pickle.dump(embeddings_dict, file, pickle.HIGHEST_PROTOCOL)
        
        else:
            print('Invalid File type')
    
    def load_embeddings(self,filepath):
        
        if (".vec" in filepath or ".txt" in filepath):
            print("Loading Model")
            f = open(filepath,'r',encoding='utf8')
            model = {}

            for line in f:
                splitLines = line.split()
                word = splitLines[0]
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                model[word] = wordEmbedding
            print(len(model)," words loaded.")

            return model
        
        elif('.plk' in filepath):
            print("Loading Model")
            f = open(filepathp,'rb',encoding='utf8')
            model = pickle.load(f)
            print(len(model.keys())," words loaded.")
            
            return model
        
        else:
            return None
    
    def concatenate_embeddings(self,embedding_dict1, embedding_dict2, intersection_only = True):
        
        embedding_num1 = len(random.choice(list(embedding_dict1.values())))
        embedding_num2 = len(random.choice(list(embedding_dict2.values()))) 
        word_set1 = set(embedding_dict1.keys())
        word_set2 = set(embedding_dict2.keys())
        print("Input Details: \nSet1: ({},{})\nSet2: ({},{})".format(len(word_set1),embedding_num1, len(word_set2),embedding_num2))
        concatenated_matrix = np.empty([0,embedding_num1 + embedding_num2])

#         count = 0
        
        if(intersection_only == True):
            vocab = word_set1.intersection(word_set2)
            print('Common Vocab:',len(vocab))
            
            for word in vocab:
                #print('Accessing word:',word)
                vec1 = embedding_dict1[word]
                vec2 = embedding_dict2[word]
                vec_conc = np.concatenate((vec1,vec2))

                concatenated_matrix = np.vstack((concatenated_matrix,vec_conc))
#                 count += 1
#                 if(count == 10): break
                    
            print('\nOutput Shape: ',(concatenated_matrix.shape))
        
        else:
            vocab = word_set1.union(word_set2)
            common_v = word_set1.intersection(word_set2)
            
            for word in vocab:
                
                if not word in common_v:
                    # Logic for projecting embeddings 
                    vec_conc = np.zeros([0,embedding_num1 + embedding_num2])
                    concatenated_matrix = np.vstack((concatenated_matrix,vec_conc))
                
                else:
                    vec1 = embedding_dict1['word']
                    vec2 = embedding_dict2['word']
                    vec_conc = np.concatenate((vec1,vec2))
                    concatenated_matrix = np.vstack((concatenated_matrix,vec_conc))
                    
        return concatenated_matrix, list(vocab)
    
    
    def perform_SVD(self,matrix, num_components = 2):
        
            print('Input Shape: {}\n'.format(matrix.shape))
            
            svd = TruncatedSVD(n_components = num_components)
            new_matrix = svd.fit_transform(matrix)
            
            print('Output Shape: {}\n'.format(new_matrix.shape))
            
            return new_matrix