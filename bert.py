# _*_ coding=utf-8 _*_
import pre  # 预处理文件
import json  # 读取json数据集
import jieba  # 分词
import csv

# 对train.json中的text计算词向量
# 读取train.json文件
with open('/home/nekozo/VSCode/#Programs/Python/Task/Diag_Nor/dataset/train.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# 将预训练数据导入字典
result_dict = {}
for item in json_data:
    result_dict[item['text']] = item['normalized_result']

# 创建列表存储未处理的数据
nonP_text = list(result_dict.keys())
nonP_result = list(result_dict.values())

# 输入待查询语句
query = "右肺有明显的'结节'，需要办理住院"

# 数据规范化(去除非中文字符)
st_text = [pre.remove_non_chinese(item) for item in nonP_text]
st_query = [pre.remove_non_chinese(query)]

# 获取停止词列表
with open('/home/nekozo/VSCode/#Programs/Python/Task/Diag_Nor/dataset/stop_word.txt', 'r') as f:
    st_words = f.readlines()
st_words = [item.strip() for item in st_words]  # 以\n分割

# 去除空值与停止词
sp_text = [item for item in st_text if item]  # remove NULL
final_train = []
for sentence in sp_text:
    p_sentence = list(jieba.cut(sentence))
    p_sentence = [item for item in p_sentence if item not in st_words]  # remove word in st_words
    text = ''.join(p_sentence)  # transfrom to string
    final_train.append(text)
sp_query = [item for item in st_query if item]  # remove NULL
cut_query = list(jieba.cut(sp_query[0]))
cut_query = [item for item in cut_query if item not in st_words]  # remove word in st_words
final_query = ''.join(cut_query)  # transfrom to string

# 调用bert模型获取文本表示
embedded_query = pre.bert_embedding(final_query)
embedded_train = []
len = len(final_train)
i = 0
for text in final_train:
    i += 1
    print(str(i) + '/' + str(len))
    embedded_text = pre.bert_embedding(text)
    embedded_train.append(embedded_text)

# 写入csv观察
fieldname = ['ID', 'DATA']
with open('train_data.csv', mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldname)
    writer.writeheader()
    i = 0
    for data in embedded_train:
        writer.writerow(
            {
                'id': i,
                'DATA': data
            }
        )
        i += 1
