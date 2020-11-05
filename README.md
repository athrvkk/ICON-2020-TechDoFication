# Technical-Domain-Identification
Submission for TechDOfication-2020 Shared Task for ICON 2020. A Machine Learning system for automatic domain identification of a given text in specified language (Marathi).

### Machine learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|Multinomial Naive Bayes|<p>BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF<br>fasttext embeddings<br>Indic-fasttext embeddings</p>|<p>86.29<br>74.92<br>65.13<br>77.53<br>-<br>-</p>|<p>0.8516<br>0.7431<br>0.5675<br>0.7440<br>-<br>-</p>|
|Linear SVC|<p>BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF<br>fasttext embeddings<br>Indic-fasttext embeddings</p>|<p>83.91<br>85.63<br>85.97<br>87.56<br>72.88<br>78.09</p>|<p>0.8282<br>0.8555<br>0.8489<br>0.8649<br>0.7000<br>0.7578</p>|


### Deep learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|FFNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>59.39<br>71.50<br>74.73</p>|<p>0.4475<br>0.6929<br>0.7189</p>|
|CNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>72.59<br>77.08<br>84.02</p>|<p>0.7024<br>0.7514<br>0.8279</p>|
|Bi-LSTM|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>80.00<br>83.12<br>87.03</p>|<p>0.7870<br>0.8215<br>0.8623</p>|
