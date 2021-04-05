#coding=utf-8
import csv
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.manifold import TSNE
from gensim.models.word2vec import Word2Vec
infile=pd.read_csv("/Users/xidyugavin/Library/Mobile Documents/com~apple~CloudDocs/G's/Prostgraduate in CityU_HK/Sememter B/COM5508 Media Data Analytics/COM5508_Coding& Data/Bias/data/cleaned.csv")
post=infile['post']
post=post.values.tolist()
post=[l.strip() for l in post]
post=[l for l in post if l !=""]
all_posts = []
for line in post:
    line_tokens=nltk.word_tokenize(line)
    line_tokens=[token.lower() for token in line_tokens if token.isalpha()]
    all_posts.append(line_tokens)
stopwords = nltk.corpus.stopwords.words('english')
filtered_posts = []
for line_tokens in all_posts:
    line_tokens=[token for token in line_tokens if token not in stopwords]
    filtered_posts.append(line_tokens)
model=Word2Vec(filtered_posts)
#model.wv.most_similar('hate')
#num_emb=1000
#words=model.wv.index_to_key[:num_emb]
#vectors=model.wv.vectors[:num_emb]
#vectors_embedded= TSNE(n_components=2).fit_transform(vectors)
#plt.figure()
#for i, word in enumerate(words):
#    plt.plot(vectors_embedded[i][0], vectors_embedded[i][1], "bs")
#    plt.text(vectors_embedded[i][0], vectors_embedded[i][1], word)
#plt.tight_layout()
#plt.show()