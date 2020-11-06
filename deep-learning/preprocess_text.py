#!/usr/bin/env python
# coding: utf-8

# In[27]:


import re
from inltk.inltk import tokenize
from inltk.inltk import identify_language
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
import pandas as pd
from nltk import pos_tag, word_tokenize
from textblob import TextBlob, Word


# In[63]:


class Preprocess:
    
        # --------------------------------------- Constructor --------------------------------------- 
        
        def __init__(self,stopword_list):
            self.data_path = ''
            self.stopword_list = stopword_list
            
    
        # --------------------------------------- Preprocess --------------------------------------- 
        
        def clean_text(self,text):
            
            special_chars = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            text = str(text)
            
            # initializing Lemmatizer and Stemmer
            lemmatizer = WordNetLemmatizer()
            stemmer = PorterStemmer() 
            
            # Cleaning the urls
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

            # Cleaning the html elements
            text = re.sub(r'<.*?>', '', text)
            
            # Removing the punctuations
            text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', ' ', text)
                    
            # Removing stop words
            text = ' '.join([word for word in text.split() if word not in self.stopword_list])
            
            # Cleaning the whitespaces
#             text = re.sub(r'\s+', '', text).strip()

            preprocessed_text = ""
            
            for word in text.split(): 
#                 if any(chr.isalpha() for chr in word) and any(chr.isdigit() for chr in word): 
                    if (re.match('\d+', word)):
                       if(word.isnumeric()):
                           preprocessed_text = preprocessed_text + '<Numeric>' + " "
                    
                    else:
                            if re.match('[a-zA-Z]+', word):
                                word = word.lower()
                                word = lemmatizer.lemmatize(word)
                                preprocessed_text = preprocessed_text + word + " "
                                
                            else:
                                preprocessed_text = preprocessed_text + word + " "
                       

            # Lemmatizing Marathi words (optional)
#             for word in text:
#                 if not re.match('[a-zA-Z]+',word) and word != '#s':
                    
            
            
            return preprocessed_text
# In[68]:


if __name__ == '__main__':
   
    #df = pd.read_csv('../Technodifacation/Data/training_data_marathi.csv')
    stopword_list = []
    
    with open ('../Technodifacation/Data/marathi_stopwords.txt','r',encoding='utf') as st:
        st_content = st.read()
        st_list = set(st_content.split())
        stopword_list = st_list

    # sample_text = df.sample()['text'].values[0]
    sample_text = "!@#@$ ! $!@ $!@$ !@$ ! $ एका विशिष्ट 19022323239 शब्दाचा उच्चार कसा केला गेला आणि 99 Working समन्वय साधण्याचा प्रयत्न करा जेव्हा 87929999 एका बिंदूबरोबर इतर गोष्टींचा विचार केला तर आपण एका चांगल्या स्थितीत जाऊ शकता!!! Google computer architecture graphic show.!!!"
    preprocessed_text = pp.clean_text(sample_text)
    print('\nBefore:\t',sample_text,'\n\nAfter:\t',preprocessed_text)


# In[ ]:




