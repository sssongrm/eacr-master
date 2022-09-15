# 展示所有词向量

import os
import sys
path_ = os.path.join(os.path.abspath('./'), 'nlp')
sys.path.append(path_)
from lstm import input_transform,loadfile,tokenizer,word2vec_train

def output_wordvector():
    
    combined,y=loadfile()
    print (len(combined),len(y))
    combined = tokenizer(combined)
    index_dict, word_vectors,combined=word2vec_train(combined)

    filebuf = open('nlp/analysis/test_word2vec/wordvector.txt','w',encoding='utf-8')

    for i in range(1,7000):
        word = list(index_dict.keys())[list(index_dict.values()).index(i)]
        print(word)
        vector = str(word_vectors[word]).replace("\n","")
        vector = vector.replace("\t","")
        print(vector)
        filebuf.write(word+" : "+vector+'\n')

if __name__ == "__main__":
    output_wordvector()