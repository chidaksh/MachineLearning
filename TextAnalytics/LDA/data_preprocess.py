import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

num_samp = 1000
data, _ = fetch_20newsgroups(shuffle=True, random_state=1, remove=(
                'headers', 'footers', 'quotes'), return_X_y=True)
data_samples = data[:num_samp]
tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, stop_words='english')
tf = tf_vectorizer.fit_transform(data_samples)
vocabulary = tf_vectorizer.vocabulary_

docs = []
for row in tf.toarray():
    present_words = np.where(row != 0)[0].tolist()
    present_words_with_count = []
    for word_idx in present_words:
        for count in range(row[word_idx]):
            present_words_with_count.append(word_idx)
    docs.append(present_words_with_count)

num_docs = len(docs)
num_words = len(vocabulary)
num_topics = 10

print("Number of documents: ",num_docs)
print("Number of words in the Vocabulary: ",num_words)
print("Total Number of Topics: ",num_topics)