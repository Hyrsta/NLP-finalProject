# _*_ coding=utf-8 _*_
import re   # 正则表达式
import jieba


# 去除停用词并tokenized句子
def delete_stop_words(input_string, stop_words_list):
    # chinese_only = re.sub('[^\u4e00-\u9fa5]', '', input_string)
    result = []
    for word in jieba.lcut(input_string):
        if word not in stop_words_list:
            result.append(word)
    return result


# test
if __name__ == '__main__':
    with open('dataset/stop_words.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = f.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)
    query = '右肺,结节转移c1c1'
    st = delete_stop_words(query, stop_words)
    print(st)
