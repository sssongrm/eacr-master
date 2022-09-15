import time
import requests
import csv
import json
import os
import random
import re

#爬虫

def filter_emoji(desstr, restr=''):
    try:
        co = re.compile(u'[\U00010000-\U0010ffff]')
    except re.error:
        co = re.compile(u'[\uD800-\uDBFF][\uDC00-\uDFFF]')
    return co.sub(restr, desstr)

def make_urlmsg(name):
    urllist = ['','','','']
    with open('nlp/analysis/comment/url_msg.csv',encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            tmpstr = str(row[0])[1:8]
            if tmpstr == name:
                for i in range(1,5):
                    urllist[i-1]=str(row[i])
    return urllist

def getcomment(urlmessage,filename):

    endpage = 40
    headers = {
    'referer': urlmessage[0],
    'user-agent': urlmessage[1],
    'cookie': urlmessage[2]
    }

    url = urlmessage[3]
    data = requests.get(url, headers=headers).text 
    pat = re.compile('"rateContent":"(.*?)","fromMall"')
    pat.findall(data)
    texts = []

    for i in range(1, endpage):
        tmpstr = "currentPage="+str(i)
        tmpulrstr = urlmessage[3].replace("currentPage=1",tmpstr)
        url = tmpulrstr
        time.sleep(random.randint(5, 10))
        data = requests.get(url, headers=headers).text
        texts.extend(pat.findall(data))
        print("第{}页评论爬取完毕".format(i))

    filepath = 'nlp/analysis/comment/'+filename+'.txt'
    fb = open(filepath, 'w', encoding='utf-8')
    for text in texts:
        tmpstr = str(filter_emoji(text))
        fb.write( tmpstr+'\n')
    print("评论保存完毕！")
    fb.close()

if __name__ == "__main__":

    msg1_macbook = make_urlmsg('macbook')
    getcomment(msg1_macbook,'macbook1')