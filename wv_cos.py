# _*_ coding=utf-8 _*_ 
import pre      # 预处理文件
import json     # 读取json数据集
import openpyxl
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
standard_disease = []
for row in xlsx_file.worksheets[0]:
    standard_disease.append(row[1].value)
# print(standard_disease)

# 将train_json数据导入字典
train_dict = {}
for item in json_data:
    train_dict[item['text']] = item['normalized_result']

# 创建列表存储未处理的数据
nonP_text = list(train_dict.keys())
nonP_result = list(train_dict.values())

# 输入待查询语句
query = "左膝退变伴游离体"

#  数据规范化
pre_text = [pre.preprocess(item, stop_words) for item in nonP_text]
for data in nonP_result:
    pre_result = [pre.preprocess(item, stop_words) for item in data.split('##')]
pre_standard = [pre.preprocess(item, stop_words) for item in standard_disease]
pre_query = pre.preprocess(query, stop_words)

# 建立WV模型并计算train_text的词向量
model = pre.w2v_model(pre_standard)
query_in_model = []
for word in pre_query:
    if word in model.wv.index_to_key:
        query_in_model.append(word)
similar_words = model.wv.most_similar(query_in_model)
correct_answer = train_dict[query]
print(pre_query)
print(query_in_model)
print(similar_words)
print(correct_answer)



# 计算文本向量
# for word in zip(pre_text, pre_query):
    # print(word)
    # sum = 0
    # if word in model.wv.index_to_key:
    #     sum +=
# query_vector = sum(model.wv[word] for word in pre_query if word in model.wv.index_to_key) / len(pre_query)
# print(query_vector)

# train_vector = []
# for i in pre_text:
#     # print(i)y
#     vec = sum(model.wv[word] for word in i) / len(i)
#     train_vector.append(vec)
# # 计算余弦相似度并排序
# similarities = [cosine_similarity([query_vector], [vector])[0][0] for vector in train_vector]
# # 输出与待查寻语句相似度最高的nonP_text的索引
# most_similar_index = similarities.index(max(similarities))

# print(nonP_text[most_similar_index])
# print(nonP_result[most_similar_index])
