# _*_ coding=utf-8 _*_
import pre      # 预处理文件
import json     # 读取json数据集
import openpyxl
import jieba
from sklearn.metrics.pairwise import cosine_similarity

# 对train.json中的text计算词向量
# 读取train.json文件和标准词xlsx文件
with open('train.json', 'r', encoding='utf-8') as file:
    json_data = json.load(file)

with open('stop_words.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
    con = f.readlines()
    stop_words = set()  # 集合可以去重
    for i in con:
        i = i.strip()  # 去掉读取每一行数据的\n
        stop_words.add(i)

# 导入xlsx文件里的标准病毒
xlsx_file = openpyxl.load_workbook("鍥介檯鐤剧梾鍒嗙被 ICD-10鍖椾含涓村簥鐗坴601.xlsx")
nonP_standard = []
for row in xlsx_file.worksheets[0]:
    nonP_standard.append(row[1].value)

# 将train_json数据导入字典
train_dict = {}
for item in json_data:
    train_dict[item['text']] = item['normalized_result']

# 创建列表存储未处理的数据
nonP_text = list(train_dict.keys())
nonP_result = list(train_dict.values())

nonP_result_split = []
for item in nonP_result:
    for sentence in item.split('##'):
        nonP_result_split.append(sentence)

#  数据规范化
pre_text = [pre.delete_stop_words(item, stop_words) for item in nonP_text]
pre_result = [jieba.lcut(item) for item in nonP_result_split]
pre_standard = [jieba.lcut(item) for item in nonP_standard]

# 建立WV模型并计算pre_standard的词向量
model = pre.w2v_model(pre_result)

# 计算pre_result句子的向量
result_vector = []
for i in pre_result:
    vec = sum(model.wv[word] for word in i if word in model.wv.index_to_key) / len(i)
    result_vector.append(vec)

for index, query in enumerate(nonP_text):
    # 待查询语句的正确结果
    correct_answer = train_dict[query]

    # 数据规范化
    pre_query = pre.delete_stop_words(query, stop_words)

    # 计算pre_query句子的向量
    query_vector = sum(model.wv[word] for word in pre_query if word in model.wv.index_to_key) / len(pre_query)

    # 计算余弦相似度并排序
    similarities = [cosine_similarity([query_vector], [vector])[0][0] for vector in result_vector]

    # 输出与待查寻语句相似度最高的nonP_text的索引
    most_similar_value = max(similarities)
    most_similar_index = similarities.index(most_similar_value)
    predicted_answer = nonP_result_split[most_similar_index]

    # 输出待查询语句，预测结果，以及正确结果
    print("")
    print("待查询语句：{} \n预测结果：{} \n正确结果：{}\n".format(query, predicted_answer, correct_answer))
