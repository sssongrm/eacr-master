import h5py
import jieba
import sys
import numpy as np
import pandas as pd
import multiprocessing
from gensim.models import Word2Vec
from gensim.corpora.dictionary import Dictionary
from tensorboard import summary
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers.core import Dense, Dropout,Activation
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python import keras
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

np.random.seed(1337)
sys.setrecursionlimit(1000000)
vocab_dim = 200
maxlen = 100
n_iterations = 5
n_exposures = 10
window_size = 7
batch_size = 32
n_epoch = 1
input_length = 100
cpu_count = multiprocessing.cpu_count()

def loadfile():
    neg=pd.read_csv('data/neg2.csv', header=None, index_col=None)
    pos=pd.read_csv('data/pos2.csv', header=None, index_col=None, error_bad_lines=False)
    neu = pd.read_csv('data/neutral.csv', header=None, index_col=None)
    combined = np.concatenate((pos[0],neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int), -1 * np.ones(len(neg), dtype=int)))
    return combined,y

def tokenizer(text):
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text

def create_dictionaries(model=None,
                        combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined
    else:
        print('No model provided...')

def show_history(history):

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Test acc')
    plt.title('Training and Test accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    plt.savefig("TrainAndLoss.jpg")

def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('model/lstm_model/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return   index_dict, word_vectors,combined

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]
    print(combined)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):

    print ('Defining a Simple Keras Model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))
    model.add(LSTM(units=vocab_dim, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))

    print ('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',metrics=['accuracy'])

    print ("Train...")
    history=model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))

    show_history(history)

    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,batch_size=batch_size)

    model.save('model/lstm_model/lstm.h5')
    model.save_weights('model/lstm_model/lstm_three.h5')
    print('Test score:', score)

def train():
    print ('Loading Data...')
    combined,y=loadfile()
    print (len(combined),len(y))
    print ('Tokenising...')
    combined = tokenizer(combined)
    print ('Training a Word2vec model...')
    index_dict, word_vectors,combined=word2vec_train(combined)
    print ('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print (x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)

def input_transform(string):
    words = jieba.lcut(string)
    print(words)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('model/lstm_model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined

def input_transform_for_analysis(string):
    words = jieba.lcut(string)
    cut_words = words
    print(words)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('model/lstm_model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined,cut_words

def load_model():
    print('loading model......')
    file = h5py.File('model/lstm_model/lstm.h5')
    model = keras.models.load_model(file)
    print('loading weights......')
    
    model.load_weights('model/lstm_model/lstm_three.h5')
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    return model

def lstm_predict_single(string):
    model = load_model()
    data = input_transform(string)
    data.reshape(1, -1)
    result = model.predict(data)
    print(result)
    if result [0][1] >= 0.5:
        print(string, ' positive')
        return 'Positive'
    elif result[0][0] >= 0.5:
        print(string, 'Neutral')
        return 'Neutral'
    else:
        print(string, ' negative')
        return 'Negative'

if __name__=='__main__':

    train()

    lstm_predict_single("蒙牛的新包装真清新！！")
