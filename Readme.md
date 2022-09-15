# eacr-master
基于LSTM的淘宝商品评论系统NLP课设，通过Django实现了Demo测试网页

### 环境
* Windows10
* Python 3.7.6
* GPU NVIDIA GeForce GTX 1050 TI

### 依赖
```
pip3 install -r requirements.txt
```

### 数据集
见`nlp/data/`路径下csv文件，分为积极、中性、消极三种。

### 运行
运行网页DEMO:
```
python3 manage.py runserver
```
训练LSTM模型：
```
cd nlp
python3 lstm.py
```
为了与LSTM模型做比较，同时训练了三个传统机器学习模型：支持向量机、随机森林、朴素贝叶斯，并测试：
```
cd nlp
python3 traditional_model.py
```
通过爬虫爬取淘宝商品评论：
```
cd nlp/analysis/
python3 crawler.py
```
绘制词云：
```
cd nlp/analysis/
python3 analysis_mywordcloud.py
```

### 参考
网页模板下载自：
>https://www.downdemo.com/
