import model
import json
import openpyxl
import pre


# 文件的开始
if __name__ == '__main__':
    # 读取停止词文件
    with open('dataset/stop_words.txt', 'r', encoding='utf-8') as file:  # 可根据需要打开停用词库，然后加上不想显示的词语
        con = file.readlines()
        stop_words = set()  # 集合可以去重
        for i in con:
            i = i.strip()  # 去掉读取每一行数据的\n
            stop_words.add(i)

    # 导入xlsx文件里的标准病毒
    xlsx_file = openpyxl.load_workbook("dataset/鍥介檯鐤剧梾鍒嗙被 ICD-10鍖椾含涓村簥鐗坴601.xlsx")
    nonP_standard = []
    for index, row in enumerate(xlsx_file.worksheets[0]):
        if row[1].value not in nonP_standard:
            nonP_standard.append(row[1].value)

    # 导入数据集
    nonP_text, nonP_result, nonP_result_list = pre.dataset_to_list('dataset/train.json')

    # 数据预处理
    pre_result = [pre.delete_punctuations(item) for item in nonP_result_list]
    pre_standard = [pre.delete_punctuations(item) for item in nonP_standard]

    # 读取pre.json文件
    with open('dataset/pre.json', 'r', encoding='utf-8') as file:
        train_json = json.load(file)

    # 由train_json创建以诊断词为键和标准词为值的字典
    train_dict = {}
    for item in train_json:
        train_dict[item['text']] = item['normalized_result']

    # 创建列表存储train_json的标准词
    nonP_result = list(train_dict.values())
    nonP_result_list = []
    for item in nonP_result:
        for sentence in item.split('##'):
            if sentence not in nonP_result_list:
                nonP_result_list.append(sentence)

    # 数据预处理
    pre_result = [pre.delete_punctuations(item) for item in nonP_result_list]
    pre_standard = [pre.delete_punctuations(item) for item in nonP_standard]

    # 建立WV模型并计算pre_standard的词向量
    my_model = model.w2v_model(pre_result)

    # 保存模型
    my_model.save("model/train.model")
