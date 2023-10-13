# _*_ coding=utf-8 _*_ 
import pre      # 预处理文件
import json     # 读取json数据集
import jieba    # 分词
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 对train.json中的text计算词向量    
# 读取train.json文件和标准词xlsx文件
with open('train.json', 'r', encoding='utf-8') as file:
    json_data_1 = json.load(file)

with open('stop_words.txt', encoding='utf-8') as f:  # 可根据需要打开停用词库，然后加上不想显示的词语
    con = f.readlines()
    stop_words = set()  # 集合可以去重
    for i in con:
        i = i.strip()  # 去掉读取每一行数据的\n
        stop_words.add(i)

df = pd.read_excel("鍥介檯鐤剧梾鍒嗙被 ICD-10鍖椾含涓村簥鐗坴601.xlsx")
# 将标准词导入字典
standard_word_list = df.iloc[:, 1].values.tolist()
# print(standard_word_list)

# 将预训练数据导入字典
result_dict = {}
for item in json_data_1:
    result_dict[item['text']] = item['normalized_result']

# 创建列表存储未处理的数据
nonP_text = list(result_dict.keys())
nonP_result = list(result_dict.values())
# print(nonP_text)
# print(len(nonP_text))

# 输入待查询语句
query = "右肺结节住院得得得fjdfk    "

#  数据规范化
sp_standard = [pre.preprocess(item, stop_words) for item in standard_word_list]
sp_text = [pre.preprocess(item, stop_words) for item in nonP_text]
sp_query = pre.preprocess(query, stop_words)
print(sp_query)
print(sp_text)

# 建立WV模型并计算train_text的词向量
model = pre.wv_model(sp_text + sp_query)

# 计算文本向量
query_vector = sum(model.wv[word] for word in sp_query if word in model.wv.index_to_key) / len(sp_query)
# print(query_vector)
train_vector = []
for i in sp_text:
    # print(i)
    vec = sum(model.wv[word] for word in i) / len(i)
    train_vector.append(vec)
# 计算余弦相似度并排序
similarities = [cosine_similarity([query_vector], [vector])[0][0] for vector in train_vector]
# 输出与待查寻语句相似度最高的nonP_text的索引
most_similar_index = similarities.index(max(similarities))

# print(nonP_text[most_similar_index])
# print(nonP_result[most_similar_index])
