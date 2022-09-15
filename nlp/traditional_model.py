import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import pandas as pd
import jieba
import joblib
from sklearn import svm
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score

def load_file_and_preprocessing():

    neg=pd.read_csv('data/neg2.csv', header=None, index_col=None)
    pos=pd.read_csv('data/pos2.csv', header=None, index_col=None, error_bad_lines=False)
    neu = pd.read_csv('data/neu.csv', header=None, index_col=None)
    combined = np.concatenate((pos[0],neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int), -1 * np.ones(len(neg), dtype=int)))
    x_train,x_test,y_train,y_test=train_test_split(combined,y,test_size=0.2)

    np.save('model/y_train.npy', y_train)
    np.save('model/y_test.npy', y_test)
    return x_train,x_test

def build_sentence_vector(text,size,w2v_model):
    vec=np.zeros(size).reshape((1,size))
    count=0
    for word in text:
        try:
            vec+=w2v_model.wv[word].reshape((1,size))
            count+=1
        except KeyError:
            continue
    if count!=0:
        vec/=count
    return vec

def get_train_vecs(x_train,x_test):
    n_dim=300
    w2v_model=Word2Vec(size=n_dim,window=5,sg=0,hs=0,negative=5,min_count=10)
    w2v_model.build_vocab(x_train)
    w2v_model.train(x_train, total_examples=w2v_model.corpus_count, epochs=5)
    train_vecs=np.concatenate([build_sentence_vector(z,n_dim,w2v_model) for z in x_train])
    np.save('model/train_vecs.npy', train_vecs)
    print('训练集的维度：',train_vecs.shape)

    w2v_model.train(x_test, total_examples=w2v_model.corpus_count, epochs=5)
    w2v_model.save('model/w2v_model/w2v_model.pkl')
    test_vecs=np.concatenate([build_sentence_vector(z,n_dim,w2v_model) for z in x_test])
    np.save('/model/test_vecs.npy', test_vecs)
    print('测试集的维度：',test_vecs.shape)

def get_data():
    train_vecs=np.load('model/train_vecs.npy')
    print('训练集向量为：',train_vecs)
    y_train=np.load('model/y_train.npy')
    print('训练集标签为：',y_train)
    test_vecs=np.load('model/test_vecs.npy')
    print('测试集向量为：',test_vecs)
    y_test=np.load('model/y_test.npy')
    print('测试集标签为：',y_test)
    return train_vecs,y_train,test_vecs,y_test

def plot_hyperplane(clf, X, y, h=0.02, draw_sv=True, title='hyperplan'):

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    plt.title(title)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap='hot', alpha=0.5)

    markers = ['o', 's', '^']
    colors = ['b', 'r', 'c']
    labels = np.unique(y)
    for label in labels:
        plt.scatter(X[y==label][:, 0], 
                    X[y==label][:, 1], 
                    c=colors[label], 
                    marker=markers[label])
    if draw_sv:
        sv = clf.support_vectors_
        plt.scatter(sv[:, 0], sv[:, 1], c='y', marker='x')

def svm_train(train_vecs,y_train,test_vecs,y_test):

    X, y = make_blobs(n_samples=100, centers=2, 
                  random_state=0, cluster_std=0.3)
    clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
    clf_rbf.fit(train_vecs,y_train)
    joblib.dump(clf_rbf, 'model/svm_model/model.pkl')

    print ('SVM-训练集准确率：', accuracy_score(clf_rbf.predict(train_vecs), y_train))
    print ('SVM-测试集准确率：', accuracy_score(clf_rbf.predict(test_vecs), y_test))
    
    print ('SVM-训练集召回率：', recall_score(clf_rbf.predict(train_vecs), y_train,average='micro'))
    print ('SVM-测试集召回率：', recall_score(clf_rbf.predict(test_vecs), y_test,average='micro'))
    
    clf_rbf.fit(X, y)
    plt.figure(figsize=(5,5), dpi=144)
    plot_hyperplane(clf_rbf, X, y, h=0.01, title='Gaussian Kernel with $\gamma=0.5$')

def bayes_train(train_vecs,y_train,test_vecs,y_test):
    clf = naive_bayes.GaussianNB()
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model/bayes_model/model.pkl')
    print ('Bayes-训练集准确率：', accuracy_score(clf.predict(train_vecs), y_train))
    print ('Bayes-测试集准确率：', accuracy_score(clf.predict(test_vecs), y_test))
    
    print ('Bayes-训练集召回率：', recall_score(clf.predict(train_vecs), y_train,average='micro'))
    print ('Bayes-测试集召回率：', recall_score(clf.predict(test_vecs), y_test,average='micro'))

def randomforest_train(train_vecs,y_train,test_vecs,y_test):
    clf = RandomForestClassifier(random_state=0)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model/randomforest_model/model.pkl')
    print ('Randomforest-训练集准确率：', accuracy_score(clf.predict(train_vecs), y_train))
    print ('Randomforest-测试集准确率：', accuracy_score(clf.predict(test_vecs), y_test))
    
    print ('Randomforest-训练集召回率：', recall_score(clf.predict(train_vecs), y_train,average='micro'))
    print ('Randomforest-测试集召回率：', recall_score(clf.predict(test_vecs), y_test,average='micro'))

def get_predict_vecs(words):
    n_dim=300
    w2v_model=Word2Vec.load('model/w2v_model/w2v_model.pkl')
    train_vecs=build_sentence_vector(words,n_dim,w2v_model)
    return train_vecs

def svm_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('model/svm_model/model.pkl')
    result=clf.predict(words_vecs)
    print(result)
    if int(result[0])==1:
        print(string,'\n','SVM预测结果为：positive')
        return "Positive"
    elif int(result[0])==-1:
        print(string,'\n','SVM预测结果为：negative')
        return "Negative"
    else:
        print(string,'\n','SVM预测结果为：neutral')
        return "Neutral"

def bayes_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('model/bayes_model/model.pkl')
    result=clf.predict(words_vecs)
    print(result)
    if int(result[0])==1:
        print(string,'\n','朴素贝叶斯模型预测结果为：positive')
        return "Positive"
    elif int(result[0])==-1:
        print(string,'\n','朴素贝叶斯模型预测结果为：negative')
        return "Negative"
    else:
        print(string,'\n','朴素贝叶斯模型预测结果为：neutral')
        return "Neutral"

def randomforest_predict(string):
    words=jieba.lcut(string)
    words_vecs=get_predict_vecs(words)
    clf=joblib.load('model/randomforest_model/model.pkl')
    result=clf.predict(words_vecs)
    print(result)
    if int(result[0])==1:
        print(string,'\n','随机森林模型预测结果为：positive')
        return "Positive"
    elif int(result[0])==-1:
        print(string,'\n','随机森林模型预测结果为：negative')
        return "Negative"
    else:
        print(string,'\n','随机森林模型预测结果为：neutral')
        return "Neutral"

if __name__=='__main__':

    x_train, x_test=load_file_and_preprocessing()
    get_train_vecs(x_train, x_test)
    train_vecs, y_train, test_vecs, y_test=get_data()

    svm_train(train_vecs, y_train, test_vecs, y_test)
    bayes_train(train_vecs, y_train, test_vecs, y_test)
    randomforest_train(train_vecs, y_train, test_vecs, y_test)

    string='太垃圾了，不怎么样'
    svm_predict(string)
    bayes_predict(string)
    randomforest_predict(string)
