import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train = pd.read_csv('training.txt', sep="\t", names=['sentiments','text'])
test = pd.read_csv('testdata.txt',skiprows=1, names=['text'], delimiter='\n')


def text_process(text):
    nopunc = ''.join([w for w in text if w not in string.punctuation])
    return [w for w in nopunc.split() if w.lower() not in stopwords.words('english')]
t = text_process(train['text'][0])

pipeline = Pipeline([
            ('bag_of_words', CountVectorizer(analyzer=text_process)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultinomialNB())
        ])

pipeline.fit(train['text'], train['sentiments'])
predictions = pipeline.predict(test['text'])
result = pd.DataFrame({
            'sentiment': predictions,
            'text': test['text']
        })

result.to_csv('result.csv', index=False)