# Technical-Domain-Identification
Submission for TechDOfication-2020 Shared Task for ICON 2020. A Machine Learning system for automatic domain identification of a given text in specified language (Marathi).

### Machine learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|Multinomial Naive Bayes|<p>BoW<br>char-BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF</p>|<p>86.74<br>81.61<br>77.16<br>61.98<br>76.93</p>|<p>0.8532<br>0.8010<br>0.7251<br>0.5138<br>0.7329</p>|
|Linear SVC|<p>BoW<br>char-BoW<br>TF-IDF<br>n-gram TF-IDF<br>character n-gram TF-IDF<br>Indic-fasttext embeddings(Bow)<br>Indic-fasttext embeddings(TF-IDF)<br>Domain-Specific Embeddings(Bow)<br>Domain-Specific Embeddings(TF-IDF)</p>|<p>85.76<br>86.19<br>88.17<br>87.27<br>88.78<br>79.20<br>77.67<br>85.44<br>85.42</p>|<p>0.8435<br>0.8467<br>0.8681<br>0.8614<br>0.8757<br>0.7691<br>0.7513<br>0.8419<br>0.8414</p>|



### Deep learning results:

|Model|Input Features|Validation Accuracy|Average F1-Score|
|:-------|:--------|:-------|:--------|
|FFNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings<br>Domain specific embeddings (raw)</p>|<p>59.39<br>71.50<br>76.42<br>76.11</p>|<p>0.4475<br>0.6929<br>0.7462<br>0.7454</p>|
|CNN|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings<br>Domain specific embeddings (raw)</p>|<p>72.59<br>77.08<br>84.20<br>86.66</p>|<p>0.7024<br>0.7514<br>0.8312<br>0.8532</p>|
|Bi-LSTM|<p>fasttext embeddings<br>Indic-fasttext embeddings<br>Domain specific embeddings<br>Domain specific embeddings (raw)</p>|<p>80.00<br>83.12<br>87.30<br>89.31</p>|<p>0.7870<br>0.8215<br>0.8629<br>0.8842</p>|
|BiLSTM-CNN|Domain specific embeddings (raw)|88.99|0.8807|
