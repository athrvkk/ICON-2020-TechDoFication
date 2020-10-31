#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import Embedding
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, concatenate
from keras.layers import Bidirectional


# In[2]:


class Models:
    
    # --------------------------------------- Constructor --------------------------------------- 
    
    def __init__(self, tokenizer, max_len):
    
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def rnn_model(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
             
        model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 13),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(4, activation='softmax')
        ])

        model.summary()
        model.compile(loss=’categorical_crossentropy’, optimizer='adam', metrics=['accuracy'])
        NUM_EPOCHS = 200
        
#         inp1 = Input(shape=(max_len,))
#         x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
#         x = Bidirectional(LSTM(256, return_sequences=True))(x)
#         x = Bidirectional(LSTM(150))(x)
#         x = Dense(128, activation="relu")(x)
#         x = Dropout(0.1)(x)
#         x = Dense(64, activation="relu")(x)
#         x = Dense(1, activation="softmax")(x)    
#         model = Model(inputs=inp1, outputs=x)

#         model.compile(loss = 'binary_crossentropy', optimizer = 'adam')
#         self.model = model


# In[ ]:




