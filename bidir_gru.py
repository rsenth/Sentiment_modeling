#Uses FastText common crawl data

########################################
## import packages
########################################

import sys, os, re, csv, codecs 
import numpy as np
import pandas as pd

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, GRU, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

import gc
from keras import backend as K
from sklearn.metrics import log_loss

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

#Set the random seed
np.random.seed(100)

#Number of folds in K-fold validation
nfold = 10 

#Batch size
train_batch_size = 128

#Location of Embedding, training and test data
path = '../input/'
EMBEDDING_FILE='/homeb/senthil/fast_text/crawl/crawl-300d-2M.vec'
TRAIN_DATA_FILE='../input/train_fasttext_crawl_cleaned.csv'
TEST_DATA_FILE='../input/test_fasttext_crawl_cleaned.csv'


#Number of dimensions of the embedding vector
embed_size = 300 
#Number of unique words used 
max_features = 166000 
#Maximum number of words in a comment used. Shorter comments
#are padded with zero and longer comments are truncated
maxlen = 150 # max number of words in a comment to use


#Read training and test data 
train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
target_train = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values


#Make sure Tensorflow uses only a fraction of the total GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.25
config.gpu_options.visible_device_list = "0"
set_session(tf.Session(config=config))


# Standard keras preprocessing, to turn each comment into a list of word indexes of equal length (with truncation or padding as needed).
def tokenize_train_test():

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    word_index = tokenizer.word_index

    return X_train, X_test, word_index 



#Return the embedding vector for words tokenized by Keras preprocessor 
def word_to_embedding(word_index):

#Read the pretrained embedding vector 
    embeddings_index = {}
    with open(EMBEDDING_FILE) as f:
        #Skip the first line, which has info about vec
        next(f)

        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:301], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings:', len(embeddings_index)) #151,250


# Use these vectors to create our embedding matrix, with random initialization 
#for words that aren't in FastText. Use the same mean and stdev of embeddings 
#the FastText has when generating the random init.

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    print(emb_mean,emb_std)


    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector

    return embedding_matrix 


#Bidirectional two GRU layers followed by two fully connected layers
def toxic_model(embedding_matrix):

    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],trainable=False)(inp)
    x = Bidirectional(GRU(150, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = Bidirectional(GRU(150, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(x)
    x = GlobalMaxPool1D()(x)
    x = Dense(75, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model 


#Use K-Fold averaging to reduce the variance of the prediction
def kfoldcv_prediction(X_train, X_test, embedding_matrix):
    
    y_test_pred_log = 0
    y_train_pred_log=0
    y_valid_pred_log = 0.0*target_train
    index_array = [x for x in range(X_train.shape[0])]
    
    for ifold in range(nfold):
        print('\n===================FOLD=',ifold)
        
        test_idx = [x for x in index_array if x%nfold==ifold]
        train_idx = [x for x in index_array if x%nfold!=ifold]
        
        X_train_cv = X_train[train_idx]
        y_train_cv = target_train[train_idx]
        X_holdout = X_train[test_idx]
        Y_holdout= target_train[test_idx]

        #define file path 
        file_path = "%s_model_weights.hdf5"%ifold

        toxic_sent_model = toxic_model(embedding_matrix)
        toxic_sent_model.fit(X_train_cv, y_train_cv, batch_size=train_batch_size, epochs=4, \
                    validation_data=(X_holdout, Y_holdout), verbose=1)
        toxic_sent_model.save_weights(file_path)

        #Accuracy on training data
        score = toxic_sent_model.evaluate(X_train_cv, y_train_cv, batch_size=512, verbose=1)
        print('Train loss:', score[0])
        print('Train accuracy:', score[1])

        #Accuracy on validation data
        score = toxic_sent_model.evaluate(X_holdout, Y_holdout, batch_size=512, verbose=1)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Prediction on validation(holdout) data
        pred_valid = toxic_sent_model.predict(X_holdout, batch_size=512, verbose=1)
        y_valid_pred_log[test_idx] = pred_valid

        #Prediction on test data 
        temp_test = toxic_sent_model.predict(X_test, batch_size=512, verbose=1)
        y_test_pred_log += temp_test

        del toxic_sent_model
        gc.collect()
        #K.clear_session()


    y_test_pred_log = y_test_pred_log/nfold

    print(' Test Log Loss Validation= ',log_loss(target_train, y_valid_pred_log))
    return y_valid_pred_log, y_test_pred_log

def main():

    X_train, X_test, word_index = tokenize_train_test()
    embedding_matrix = word_to_embedding(word_index)
    y_train_pred, y_test = kfoldcv_prediction(X_train, X_test, embedding_matrix)

    cwd = os.getcwd()
    fname = cwd.split('/')[-1] + '.csv'

    sample_submission = pd.read_csv('../input/sample_submission.csv')
    sample_submission[list_classes] = y_test
    sample_submission.to_csv(fname, index=False)

    train_pred = train.copy()
    train_pred.drop('comment_text',axis=1,inplace=True)
    train_pred[list_classes] = y_train_pred
    train_pred.to_csv('train_'+fname, index=False) 



if __name__ == "__main__":
    main()
