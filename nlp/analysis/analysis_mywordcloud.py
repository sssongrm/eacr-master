#!/usr/bin/python
# -*- coding: UTF-8 -*-

#画词云

def wordcloud_file(filename,inputnum):
    import jieba
    import re
    import matplotlib.pyplot as plt
    import pandas as pd  # 导入Pandas
    from matplotlib import pyplot as plt
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import jieba
    import numpy as np
    from PIL import Image
    # 待分词的文本路径
    sourceTxt = str(filename)
    print("source:{}".format(sourceTxt))
    # 分好词后的文本路径
    targetTxt = 'nlp/analysis/mywordcloud/target.txt'

    # 对文本进行操作
    with open(sourceTxt, 'r', encoding='utf-8') as sourceFile, open(targetTxt, 'w', encoding='utf-8') as targetFile:
        for line in sourceFile:
            seg = jieba.cut(line.strip(), cut_all=False)
            # 分好词之后之间用/隔断
            output = '/'.join(seg)
            targetFile.write(output)
            targetFile.write('\n')
        print('写入成功！')
        sourceFile.close()
        targetFile.close()

    txt = open("nlp/analysis/mywordcloud/target.txt", "r", encoding='utf-8').read()  # 打开txt文件,要和python在同一文件夹
    # print(txt)
    txt00 = open("nlp/analysis/mywordcloud/shuchu.txt", "w", encoding='utf-8')
    words = jieba.lcut(txt)  # 精确模式，返回一个列表
    # print(words)
    counts = {}  # 创建字典
    lt = ['三炮', '##', '......', '24', '10', '30', '2020', '14', '31', '11', '13', '20', '15', '28', '17', '16',
            '29', '微博']
    stopkeyword = [line.strip() for line in open('nlp/analysis/mywordcloud/stopwords.txt', encoding='utf-8').readlines()]  # 加载停用词
    for word in words:
        if len(word) == 1:
            continue
        elif word in stopkeyword:
            rword = " "
        else:
            rword = word
        counts[rword] = counts.get(rword, 0) + 1  # 字典的运用，统计词频
    items = list(counts.items())  # 返回所有键值对
    print(items)
    items.sort(key=lambda x: x[1], reverse=True)  # 降序排序
    # N = eval(input("请输入N：代表输出的数字个数："))
    N = inputnum
    wordlist = list()
    r1 = re.compile(r'\w')  # 字母，数字，下划线，汉字
    r2 = re.compile(r'[^\d]')  # 排除数字
    r3 = re.compile(r'[\u4e00-\u9fa5]')  # 中文
    r4 = re.compile(r'[^_]')  # 排除_
    # stopkeyword = [line.strip() for line in open('stopwords.txt', encoding='utf-8').readlines()]  # 加载停用词
    for i in range(N):
        word, count = items[i]
        txt00.write("{0:<10}{1:<5}".format(word, count))  # 输出前N个词频的词语
        txt00.write('\n')
        if r1.match(word) and r2.match(word) and r3.match(word) and r4.match(word):
            continue
    txt00.close()

    txt = open("nlp/analysis/mywordcloud/shuchu.txt", "r", encoding='utf-8').read()  # 打开txt文件,要和python在同一文件夹
    words = jieba.lcut(txt)  # 精确模式，返回一个列表
    counts = {}  # 创建字典
    for word in words:
        if len(word) == 1:  # 把意义相同的词语归一
            continue
        elif word == "三炮" or  word == "#" or word== "##" or word=="24" or word=="RAP" or word=="video":
            rword = " "
        else:
            rword = word
        counts[rword] = counts.get(rword, 0) + 1  # 字典的运用，统计词频
    items = list(counts.items())  # 返回所有键值对P168
    items.sort(key=lambda x: x[1], reverse=True)  # 降序排序
    # N = eval(input("请输入N：代表输出的数字个数")) # 这里输入300就行，因为shuchu01.txt里面的数据有限
    N = inputnum
    wordlist = list() # 创建列表并赋值
    for i in range(N):
        word, count = items[i]
        #   print("{0:<10}{1:<5}".format(word, count))  # 输出前N个词频的词语
        wordlist.append(word)  # 把词语word放进一个列表
    wl = ' '.join(wordlist)  # 把列表转换成str wl为str类型，所以需要转换
    cloud_mask = np.array(Image.open("nlp/analysis/mywordcloud/love.jpg"))  # 词云的背景图，需要颜色区分度高

    wc = WordCloud(
        background_color="white",  # 背景颜色
        mask=cloud_mask,  # 背景图cloud_mask
        max_words=100,  # 最大词语数目
        font_path='nlp/analysis/mywordcloud/simsun.ttf',  # 调用font里的simsun.tff字体，需要提前安装
        height=1200,  # 设置高度
        width=1600,  # 设置宽度
        max_font_size=1000,  # 最大字体号
        random_state=1000,  # 设置随机生成状态，即有多少种配色方案
    )

    myword = wc.generate(wl)  # 用 wl的词语 生成词云
    # 展示词云图
    plt.imshow(myword)
    plt.axis("off")
    # plt.show()
    tmp = filename.replace(".txt","")
    # tmp = "images\\"+tmp
    tmp = tmp.replace('comment','mywordcloud')
    wc.to_file(tmp+'.jpg')  # 把词云保存下当前目录（与此py文件目录相同）

if __name__ == "__main__":

    wordcloud_file('nlp/analysis/comment/macbook1.txt',100)