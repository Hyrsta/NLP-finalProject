# _*_ coding=utf-8 _*_
import pre      # 读取预处理文件
import model    # 读取模型文件
import json     # 读取json数据集
import jieba
import torch
import openpyxl
from gensim.models import Word2Vec


# 文件的开始
if __name__ == '__main__':
    # 查看是否可用cuda
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # 读取停止词文件
    with open('dataset/stop_words.txt', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 读取train.json文件
    with open('dataset/dev.json', 'r', encoding='utf-8') as file:
        train_json = json.load(file)

    # 由train_json创建以诊断词为键和标准词为值的字典
    train_dict = {}
    for item in train_json:
        train_dict[item['text']] = item['normalized_result']

    # 创建两个列表，分别存储train_json的诊断词和标准词
    nonP_text = list(train_dict.keys())
    nonP_result = list(train_dict.values())
    nonP_result_list = []
    for item in nonP_result:
        for sentence in item.split('##'):
            if sentence not in nonP_result_list:
                nonP_result_list.append(sentence)

    #  数据预处理
    pre_text = [pre.delete_stop_words(item, stop_words) for item in nonP_text]
    pre_train = [jieba.lcut(item) for item in nonP_result_list]

    # 导入模型
    my_model = Word2Vec.load("model/standard.model")
    print(my_model)

    # 将模型里的所有词放入vocabulary变量里
    vocabulary = my_model.wv.index_to_key

    # 计算pre_train句子的向量
    result_vector = model.count_sentence_vector(pre_train, my_model, vocabulary)

    # 计算pre_text句子的向量
    text_vector = model.count_sentence_vector(pre_text, my_model, vocabulary)

    # 通过余弦相似度计算对text_vector的预测答案
    predictions = model.sentence_cosine_similarity(text_vector, result_vector,
                                                   nonP_result_list, nonP_text, nonP_result, percentage=10, distance=0, max_ans=1)

    # 保存预测答案
    with open('prediction/dev_single_Mstandard.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=4)
