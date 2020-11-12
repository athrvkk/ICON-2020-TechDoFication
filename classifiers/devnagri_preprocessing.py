import pandas as pd
import numpy as np
import random
import unicodedata
from queue import Queue
from indicnlp.tokenize.indic_tokenize import trivial_tokenize_indic
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_preprocessing import *


class devnagri_preprocessing:
        
        # --------------------------------------- Constructor --------------------------------------- 
        
        def __init__(self, vowels = True, trivial_split = False):
                
                self.vowels = vowels
                self.trivial_split = trivial_split
                       
                
        # --------------------------------------- Word Splitter --------------------------------------- #
    
        def split_devanagari_word(self,word: str) -> str:
            try:
                q = Queue()
                l_index = 0
                
                if not(isinstance(word, str)): word = str(word)
                tokens = []
                
                if self.trivial_split == True:
                    tokens = [char for char in word]
                    return tokens
                
                for char in word:

                    if not 'devanagari' in unicodedata.name(char).lower():
                        tokens.append(char)
                        continue

                    if not 'sign' in unicodedata.name(char).lower():
                        if q.empty():
                            tokens.append(char)
                        else:
                            while not q.empty():
                                tokens[len(tokens)-1] += q.get() 
                            tokens.append(char)   
                    else:
                        if self.vowels == True:
                            q.put(char)

                for i, char in reversed(list(enumerate(tokens.copy()))):
                    if('devanagari' in unicodedata.name(char).lower()):
                        l_index = i
                        break

                while not q.empty():
                        tokens[l_index] += q.get() 

                return tokens

            except Exception as e:
                return ''


        # --------------------------- String to character-sequence converter --------------------------------------- #

        def text2characters(self,text:str)->str:
            try:
                if not(isinstance(text, str)): text = str(text)
                char_sequence = ""
                char_list = []
                
                for word in text.split():
                    seq = ' '.join([char for char in self.split_devanagari_word(word)])                
                    char_sequence = char_sequence + seq + ' '
                    
                return char_sequence
            
            except ValueError as ve:
                print('Error processing:\t',text)
                return ''
            
            
    # ------------------------------------- Tokenize a document --------------------------------------------- #
        
        """
            This function builds a vocabulary of each unique token (character) from the given document.
            Each token from the vocabulary is assigned a unique integer id.
            
            Working of this funcion is similar to keras.preprocess.tokenizer.fit_to_text()
        """
        
        def tokenize_characters(self, document):
            vocab = set()
            cnt = 0
            token_dict = {}
            
            if isinstance(document, list):
                for text in document:
                    char_sequence = self.text2characters(text)
                    tokens_indic = pd.Series(trivial_tokenize_indic(char_sequence))
                    word_counts = tokens_indic.value_counts()
                    
                    vocab = vocab.union(set(word_counts.keys()))

                print('Total Unique Tokens (Characters): {}'.format(len(vocab)))

                for char in vocab:
                    cnt += 1
                    token_dict[char] = cnt
            
            else:
                char_sequence = self.text2characters(document)
                tokens_indic = pd.Series(trivial_tokenize_indic(char_sequence))
                word_counts = tokens_indic.value_counts()  
                vocab = vocab.union(set(word_counts.keys()))

                print('Total Unique Tokens (Characters): {}'.format(len(vocab)))

                for char in vocab:
                    cnt += 1
                    token_dict[char] = cnt
                
            return token_dict

        
    # -------------------------Text-to-sequence converter --------------------------------------- #
        """
            This function converts the input sentence into sequence of integers. 
            Each integer corresponds to the unique token (or character) id in the vocabulary.

            Working of this funcion is similar to keras.preprocess.tokenizer.text_to_sequence()
        """

        def text_to_sequence(self,document,token_dict, pad_len=300, padding_type='post', truncating_type='post'):
            
            sequence_doc = []
            if isinstance(document, list):
                cnt = 0
                for text in document:
                    try:
                        char_array = self.text2characters(text).split()
                        text_sequence = [token_dict[x] for x in char_array]
                        sequence_doc.append(text_sequence)
                        cnt+=1
                    except:
                        print(text)
                        
                print('Records converted: ',cnt)
                
            else:
                char_array = self.text2characters(document).split()
                text_sequence = [token_dict[x] for x in char_array]
                sequence_doc.append(text_sequence)
            
            sequence_doc = np.asarray(pad_sequences(sequence_doc, 
            					    padding=padding_type, 
                                            	    truncating=truncating_type, 
                                                    maxlen=pad_len))    
            return sequence_doc


