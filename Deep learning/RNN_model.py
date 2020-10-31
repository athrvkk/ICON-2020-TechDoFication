# Deep learning: 
from keras.models import Input, Model
from keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, LSTM, Embedding,Dropout,SpatialDropout1D,Conv1D,MaxPooling1D,GRU,BatchNormalization
from tensorflow.keras.layers import Input,Bidirectional,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate,LeakyReLU

class RnnModel():
    """
    A recurrent neural network for semantic analysis
    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, X_additional=None):
        
        inp1 = Input(shape=(max_len,))
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix],input_length = max_len, trainable = False)(inp1)
        x = LSTM(256, return_sequences=True)(x)
        x = LSTM(128)(x)
        x = Dropout(0.1)(x)
        x = Dense(64, activation="softmax")(x)
        x = Dense(4, activation="softmax")(x)    
        model = Model(inputs=inp1, outputs=x)

        model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam',metrics = ['accuracy'])
        self.model = model

#         model = Sequential()
#         model.add(
#         Embedding(input_dim=embedding_matrix.shape[0],
#         output_dim=embedding_matrix.shape[1],
#         weights=[embedding_matrix],
#         input_length=max_len,
#         trainable=False))
#         model.add(SpatialDropout1D(0.5))
#         model.add(Conv1D(kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(MaxPooling1D(pool_size=2))
#         model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
#         model.add(SpatialDropout1D(0.5))
#         model.add(Conv1D(kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(MaxPooling1D(pool_size=2))
#         model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
#         model.add(SpatialDropout1D(0.5))
#         model.add(Conv1D(kernel_size=kernel_size,kernel_regularizer=regularizers.l2(0.00001), padding='same'))
#         model.add(LeakyReLU(alpha=0.2))
#         model.add(MaxPooling1D(pool_size=2))
#         model.add(Bidirectional(LSTM(lstm_units,dropout=0.5, recurrent_dropout=0.5)))
#         model.add(Dense(4,activation='softmax'))
#         model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])