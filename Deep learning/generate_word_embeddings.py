#!/usr/bin/env python
# coding: utf-8

# In[5]:


import fasttext.util
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# In[6]:


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


# In[7]:


class Embeddings:

    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    
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


# In[ ]:




