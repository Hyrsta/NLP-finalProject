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

# 输入待查询语句
query = "左膝退变伴游离体"
correct_answer = train_dict[query]
#  数据规范化
pre_text = [pre.delete_stop_words(item, stop_words) for item in nonP_text]
pre_result = []
for data in nonP_result:
    for item in data.split('##'):
        result = pre.delete_stop_words(item, stop_words)
        pre_result.append(result)
# print(pre_result)
pre_standard = [jieba.lcut(item) for item in nonP_standard]
pre_query = pre.delete_stop_words(query, stop_words)

# 建立WV模型并计算pre_standard的词向量
model = pre.w2v_model(pre_standard)
query_in_model = []
for word in pre_query:
    if word in model.wv.index_to_key:
        query_in_model.append(word)
similar_words = model.wv.most_similar(query_in_model)
print(pre_query)
print(query_in_model)
print(similar_words)
print(correct_answer)
