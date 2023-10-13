# _*_ coding=utf-8 _*_
import re   # 正则表达式
import jieba
from gensim.models import Word2Vec


# 去除非汉字和停用词
def preprocess(input_string, stop_words_list):
    chinese_only = re.sub('[^\u4e00-\u9fa5]', '', input_string)
    result = []
    for word in jieba.lcut(chinese_only):
        if word not in stop_words_list:
            result.append(word)
    return result


# 训练WV模型函数
def wv_model(text):
    model = Word2Vec(sentences=text, vector_size=300, window=10, min_count=1, workers=4)
    return model


# test
if __name__ == '__main__':
    with open('stop_words.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)
    query = '右肺,结节转移''"可能大"'
    st = preprocess(query, stop_words)
    sp = list(jieba.cut(st))
    test_model = wv_model(sp)