#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re


# In[7]:


class Preprocess:
    
        # --------------------------------------- Constructor --------------------------------------- 
        
        def __init__(self,stopword_list):
            self.data_path = ''
            self.stopword_list = stopword_list
        
        # --------------------------------------- Preprocess --------------------------------------- 
        
        def clean_text(self,string):
            
            punctuations = r'''!()-[]{};:'"\,<>./?@#$%^&*_~''',
            string = str(string)
        
            # Cleaning the urls
            string = re.sub(r'https?://\S+|www\.\S+', '', string)

            # Cleaning the html elements
            string = re.sub(r'<.*?>', '', string)
                
            # Converting the text to lower
#             string = string.lower()
            
            # Removing the punctuations
            for x in string: 
                if x in punctuations: 
                    string = string.replace(x, "")
                    
            # Removing stop words
            string = ' '.join([word for word in string.split() if word not in self.stopword_list])
            
            # Cleaning the whitespaces
            string = re.sub(r'\s+', ' ', string).strip()
            
            return string 

