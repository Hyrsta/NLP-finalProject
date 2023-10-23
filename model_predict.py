# _*_ coding=utf-8 _*_
import pre      # 读取预处理文件
import model    # 读取模型文件
import json     # 读取json数据集
from gensim.models import Word2Vec


# 文件的开始
if __name__ == '__main__':
    # 查看是否可用cuda
    device = pre.check_cuda()

    # 读取停止词文件
    with open('dataset/stop_words.txt', 'r', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 导入数据集
    dataset_path = 'dataset/dev.json'
    nonP_text, nonP_result, nonP_result_list = pre.dataset_to_list(dataset_path)
    print("successfully load", dataset_path)

    # 数据预处理
    # 删除诊断词里的停用词并tokenize句子
    pre_text = [pre.delete_stop_words(item, stop_words) for item in nonP_text]

    # 删除标准词里的标点符号并tokenize句子
    pre_train = [pre.delete_punctuations(item) for item in nonP_result_list]

    # 导入模型
    my_model = Word2Vec.load("model/train.model")
    print("successfully loading model")
    print("window:{}, epochs:{}, vector size:{}".format(my_model.window, my_model.epochs, my_model.vector_size))

    # 计算pre_train句子的向量
    result_vector = model.count_sentence_vector(pre_train, my_model)

    # 计算pre_text句子的向量
    text_vector = model.count_sentence_vector(pre_text, my_model)
    print("successfully creating sentence vectors")

    # 通过余弦相似度计算对text_vector的预测答案
    predictions = model.sentence_cosine_similarity(text_vector, result_vector, nonP_result_list,
                                                   nonP_text, nonP_result, percentage=10, distance=0, max_ans=1)

    if len(nonP_text) == len(predictions):
        save_list = []
        for data, prediction in zip(nonP_text, predictions):
            save_list.append({"text": data, "normalized_result": prediction})
    else:
        print("data and prediction not the same length")

    # 保存预测答案
    with open('prediction/dev_single_Mtrain_v11.json', 'w', encoding='utf-8') as f:
        json.dump(save_list, f, ensure_ascii=False, indent=4)
