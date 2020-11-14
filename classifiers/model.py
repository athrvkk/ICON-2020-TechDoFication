import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Bidirectional
import matplotlib.pyplot as plt

# ----------------------------------------- ML Classifier -----------------------------------------
    
def ml_classifier_model(model, x_train, x_val, y_train, y_val):
    model.fit(x_train, y_train)
    results = model.predict(x_val)
    return model, results



# ------------------------------------------------------ CALLBACK CLASS --------------------------------------------

class myCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, metrics, threshold):
        self.metrics = metrics
        self.threshold = threshold
        
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get(self.metrics) >= self.threshold):
            print("\nTraining accuracy above {}%...so training stopped...\n".format(self.threshold*100))
            self.model.stop_training = True


            
# ------------------------------------------------------ TRAIN DNN --------------------------------------------

def create_model_DNN(input_dim, embedding_dim, embedding_matrix, pad_len, trainable=False, n1=64, n2=32, kr=None, br=None):
    Simple = Sequential()
    embedding_layer = Simple.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                  trainable=trainable))
    Simple.add(GlobalMaxPooling1D())
    Simple.add(Dense(n1, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    Simple.add(Dense(n2, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    Simple.add(Dense(4, activation='softmax'))
    Simple.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
    return Simple



# ------------------------------------------------------ TRAIN CNN --------------------------------------------

def create_model_CNN(input_dim, embedding_dim, embedding_matrix, pad_len, trainable, n1, k, d=0.0, kr=None, br=None):
    myCNN = Sequential()
    myCNN.add(Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                        trainable=trainable))
    myCNN.add(Conv1D(n1, kernel_size=k, kernel_regularizer=kr, bias_regularizer=br, activation='relu'))
    myCNN.add(GlobalMaxPooling1D())
    myCNN.add(Dropout(d))
    myCNN.add(Dense(4, activation='softmax'))
    myCNN.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
    return myCNN



# ------------------------------------------------------ TRAIN RNN --------------------------------------------

def create_model_LSTM(input_dim, embedding_dim, embedding_matrix, pad_len, trainable, n1, n2, d=0.0):
    myLSTM = Sequential()
    myLSTM.add(
        Embedding(input_dim=input_dim, output_dim=embedding_dim, weights=[embedding_matrix], input_length=pad_len,
                  trainable=trainable))
    myLSTM.add(Bidirectional(LSTM(n1, dropout=d, return_sequences=True)))
    myLSTM.add(GlobalMaxPooling1D())
    myLSTM.add(Dense(n2, activation='relu'))
    myLSTM.add(Dense(4, activation='softmax'))
    myLSTM.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=['acc'])
    return myLSTM



# ----------------------------------------------- PLOT ACCURACY LOSS GRAPH --------------------------------------------

def plot_curves(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

    
    
# ------------------------------------------------------ SAVE MODELS --------------------------------------------

def save_model(model, name):
    model.save("../models/"+name+".h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open("../models/"+name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../models/"+name+"_weights.h5")

    
    
# ------------------------------------------------------ LOAD MODELS --------------------------------------------

def model_load(path):
    return load_model(path)



# ------------------------------------------------------ CLASSIFICATION REPORT --------------------------------------------
 
def classification_report(y_test, results):
    acc = accuracy_score(y_test, results)
    precision = precision_score(y_test, results, average=None)
    recall = recall_score(y_test, results, average=None)
    f1 = f1_score(y_test, results, average=None)
    return acc, precision, recall, f1



# ------------------------------------------------------ CONFUSION MATRIX --------------------------------------------

def plot_confusion_matrix(y_val, 
                          results,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_val,results)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
    
# ------------------------------------------------------ PREDICT LABELS --------------------------------------------

def predict_label(result):
    if result[0][0] >= 0.5:
        label='positive'
    else:
        label="negative"
    return label, result[0][0]
