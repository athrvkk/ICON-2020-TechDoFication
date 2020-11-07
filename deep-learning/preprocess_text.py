#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import unicodedata



# In[7]:

class Preprocess:
        
        # --------------------------------------- Constructor --------------------------------------- 
        
        def __init__(self,stopword_list):
            self.data_path = ''
            self.stopword_list = stopword_list
                

        # --------------------------------------- Preprocess --------------------------------------- 
        
        def expand_concatenations(self, word):
            
            if not re.match('[a-zA-Z]+', word) or re.match('/d+',word):
                for i in range(len(word)):
                    if not('DEVANAGARI ' in unicodedata.name(word[i])):
                        word = word[:i] if( len(word[i:]) < 2 ) else word[:i] + " " + word[i:]
                        break
            else:
                for i in range(len(word)):
                    if ('DEVANAGARI ' in unicodedata.name(word[i])):
                        word = word[i:] if( len(word[:i]) < 2 ) else word[:i] + " " + word[i:]
                        break

            return(word)
    
        
        def clean_text(self,text: str) -> str:
            try:
                special_chars = r'''!()-[]{};:'"\,<>./?@#$%^&*_~'''
                stemmer = PorterStemmer()
                lemmatizer = WordNetLemmatizer()

                if not(isinstance(text, str)): text = str(text)

                #Removing unprintable characters
                text = ''.join(x for x in text if x.isprintable())

                # Cleaning the urls
                text = re.sub(r'https?://\S+|www\.\S+', '', text)

                # Cleaning the html elements
                text = re.sub(r'<.*?>', '', text)

                # Removing the punctuations
                text = re.sub('[!#?,.:";-@#$%^&*_~<>()-]', ' ', text)


                # Removing stop words
                text = ' '.join([word for word in text.split() if word not in self.stopword_list])

                # Expanding noisy concatenations (Eg: algorithmआणि  -> algorithm आणि ) 
                text = ' '.join([self.expand_concatenations(word) for word in text.split()])

                preprocessed_text = ""

                for word in text.split(): 
                    if (re.match('\d+', word)):
                        if(word.isnumeric()):
                            preprocessed_text = preprocessed_text + '#N' + " "

                    else:
                        if(re.match('[a-zA-Z]+', word) and len(word) > 1):
                                word = word.lower()
    #                             word = lemmatizer.lemmatize(word, pos='v')
                                preprocessed_text = preprocessed_text + word + " "

                        else:
                            preprocessed_text = preprocessed_text + word + " "

                return preprocessed_text
        