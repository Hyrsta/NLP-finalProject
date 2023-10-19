from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

# Sample Chinese text
chinese_text = ["我喜欢学习自然语言处理。自然语言处理很有趣。", "你喜欢学习自然语言处理吗"]

# Tokenize the text using Jieba
tokens = [jieba.lcut(item)for item in chinese_text]
preprocessed_text = [" ".join(item) for item in tokens]
print(preprocessed_text)
# Create and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words=None)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_text)
# print(type(tfidf_matrix))
# print(tfidf_matrix)
feature_names = tfidf_vectorizer.get_feature_names_out().tolist()
# print(type(feature_names))
print(feature_names)

tfidf_score = tfidf_matrix[0, feature_names.index('喜欢')]
# print(tfidf_score)


# Get the TF-IDF scores
tfidf_scores = tfidf_matrix.toarray()
# print(type(tfidf_scores))
# print(tfidf_scores)
val = [list(zip(feature_names, item)) for item in tfidf_scores]
print(val)
