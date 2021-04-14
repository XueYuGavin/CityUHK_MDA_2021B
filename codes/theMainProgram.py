#coding=utf-8
import re
import csv
import jieba
import keras
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Activation, Input, MaxPooling1D, Flatten, concatenate, Embedding, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import nltk
import ssl
import numpy as np
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('wordnet')
nltk.download('stopwords')
ofs0=pd.read_csv("/Users/xidyugavin/Library/Mobile Documents/com~apple~CloudDocs/G's/Prostgraduate in CityU_HK/Sememter B/COM5508 Media Data Analytics/COM5508_Coding& Data/Bias/data/offensiveYN=0.csv")
ofs5=pd.read_csv("/Users/xidyugavin/Library/Mobile Documents/com~apple~CloudDocs/G's/Prostgraduate in CityU_HK/Sememter B/COM5508 Media Data Analytics/COM5508_Coding& Data/Bias/data/offensive=.5.csv")
ofs1=pd.read_csv("/Users/xidyugavin/Library/Mobile Documents/com~apple~CloudDocs/G's/Prostgraduate in CityU_HK/Sememter B/COM5508 Media Data Analytics/COM5508_Coding& Data/Bias/data/offensiveYN=1.csv")
ofs=pd.concat([ofs0,ofs5,ofs1],axis=0)
pd.set_option('display.max_colwidth', None)
stop_words=stopwords.words('english')
def remove_noise(text):
    text=' '.join(x.lower() for x in text.split())
    text=' '.join(x.strip() for x in text.split())
    text=re.sub(r'[^\w\s]', '',text)
    text=re.sub(r'[\d+\_+]', '',text)
    text=' '.join([word for word in text.split() if word not in stop_words])
    return text
ofs['process_post']=ofs['post'].apply(remove_noise)
text=ofs["process_post"]
labels=ofs["offensiveYN"]
MAX_FEATURES=5000
MAX_DOCUMENT_LENGTH=10
BATCH_SIZE=32
NUM_CLASSES=3
EPOCH=10
tokenizer=Tokenizer(num_words=MAX_FEATURES, lower=True)
tokenizer.fit_on_texts(text)
sequences=tokenizer.texts_to_sequences(text)
X=pad_sequences(sequences, maxlen=MAX_DOCUMENT_LENGTH)
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(labels)
X_train, X_test,y_train, y_test=train_test_split(X, y, test_size=0.1, random_state=123)
embedding_dims=50
filters=250
kernel_size=3
hidden_dims=250
model=Sequential()
model.add(Embedding(MAX_FEATURES, embedding_dims))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=10,
          validation_data=(X_test, y_test))
loss, accuracy=model.evaluate(X_test, y_test,batch_size=BATCH_SIZE)
#print("loss: {}, accuracy:{}".format(loss, accuracy))
embedding_dims=50
filters=250
kernel_size=3
hidden_dims=250
model=Sequential()
model.add(Embedding(MAX_FEATURES, embedding_dims))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=10,
          validation_data=(X_test, y_test))
loss, accuracy=model.evaluate(X_test, y_test,batch_size=BATCH_SIZE)
#print("loss: {}, accuracy:{}".format(loss, accuracy))
embedding_dims=50
filters=100
input=Input(shape=[MAX_DOCUMENT_LENGTH])
x=Embedding(MAX_FEATURES, embedding_dims)(input)
convs=[]
for filter_size in [3,4,5]:
    l_conv=Conv1D(filters=filters, kernel_size=filter_size, activation='relu')(x)
    l_pool=MaxPooling1D()(l_conv)
    l_pool=Flatten()(l_pool)
    convs.append(l_pool)
merge=concatenate(convs, axis=1)
out=Dropout(0.5)(merge)
output=Dense(32, activation='relu')(out)
output=Dense(units=NUM_CLASSES, activation='softmax')(output)
model=Model([input], output)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train,
          batch_size=BATCH_SIZE,
          epochs=10,
          validation_data=(X_test, y_test))
loss, accuracy=model.evaluate(X_test, y_test,batch_size=BATCH_SIZE)
#print("loss: {}, accuracy:{}".format(loss, accuracy))
predict_result=model.predict(X_test, batch_size=BATCH_SIZE)
y_pred=np.argmax(predict_result,axis=1)
#print(confusion_matrix(y_test, y_pred))
#print(classification_report(y_test, y_pred))
accuracy_score(y_test, y_pred)
with open("/Users/xidyugavin/Library/Mobile Documents/com~apple~CloudDocs/G's/Prostgraduate in CityU_HK/Sememter B/COM5508 Media Data Analytics/COM5508_Coding& Data/Bias/data/test/test2.csv") as testfile:
    reader=csv.reader(testfile)
    input_text=[row[0] for row in reader]
text=[remove_noise(i) for i in input_text]
sequences=pad_sequences(tokenizer.texts_to_sequences(text), maxlen=MAX_DOCUMENT_LENGTH)
class_=label_encoder.inverse_transform(np.argmax(model.predict(sequences), axis=-1))
print("input_text: {}, offensive classification: {}".format(input_text, class_))