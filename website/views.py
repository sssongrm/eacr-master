from django.shortcuts import render,HttpResponse
from website.models import Proposal
import datetime
# Create your views here.

def index(request):

    if request.POST.get('search_button')=='yes':
        return HttpResponse("欢迎使用")

    if request.POST.get('proposal_button')=='yes' and request.POST.get('proposal_text')!="":
        text1 = request.POST.get('proposal_text')
        time1 = datetime.datetime.now()
        Proposal.objects.create(text = str(text1),time = str(time1))
        print("写入成功！text={},time={}".format(text1,time1))

    return render(request,"index.html")

def demo(request):

    import sys
    sys.path.append("..")
    from nlp.lstm import load_model, create_dictionaries
    from gensim.models import Word2Vec
    import jieba
    import numpy as np
    model_lstm = load_model()
    model_word2vec = Word2Vec.load('nlp/model/lstm_model/Word2vec_model.pkl')

    if request.POST.get('test_button') == 'yes':
        sentence = request.POST.get('test_sentence')

        words = jieba.lcut(sentence)
        words = np.array(words).reshape(1, -1)
        _, _, data = create_dictionaries(model_word2vec, words)
        data.reshape(1, -1)
        predict_result = model_lstm.predict(data)

        if predict_result[0][1] >= 0.5:
            result='Positive'
        else :
            result='Negative'

        post_buf1 = 'The Predict Result Of \"'+sentence+'\" By LSTM Is '+ result +' !'
        return render(request,"demo.html",{'result':post_buf1})

    return render(request, "demo.html")