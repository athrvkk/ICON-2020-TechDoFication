# Technical-Domain-Identification
Submission for TechDOfication-2020 Shared Task for ICON 2020. A Machine Learning system for automatic domain identification of a given text in specified language (Marathi).

### Machine learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|Multinomial Naive Bayes|<p>BoW<br>char-BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF<br>fasttext embeddings<br>Indic-fasttext embeddings</p>|<p>86.21<br>80.66<br>78.49<br>65.02<br>77.48<br>-<br>-</p>|<p>0.8508<br>0.7938<br>0.7445<br>0.5657<br>0.7448<br>-<br>-</p>|
|Linear SVC|<p>BoW<br>char-BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF<br>Indic-fasttext embeddings(Bow)<br>Indic-fasttext embeddings(TF-IDF)<br>Domain-Specific Embeddings(Bow)<br>Domain-Specific Embeddings(TF-IDF)</p>|<p>83.99<br>84.52<br>86.69<br>86.03<br>86.93<br>77.75<br>77.35<br>84.57<br>84.94</p>|<p>0.8292<br>0.8310<br>0.8577<br>0.8506<br>0.8594<br>0.7539<br>0.7434<br>0.8316<br>0.8377</p>|



### Deep learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|FFNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>59.39<br>71.50<br>76.42</p>|<p>0.4475<br>0.6929<br>0.7462</p>|
|CNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>72.59<br>77.08<br>84.20</p>|<p>0.7024<br>0.7514<br>0.8312</p>|
|Bi-LSTM|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings</p>|<p>80.00<br>83.12<br>87.30</p>|<p>0.7870<br>0.8215<br>0.8629</p>|
