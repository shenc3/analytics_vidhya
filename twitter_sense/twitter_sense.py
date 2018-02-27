import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


# Questions:
#   1. 数据不平衡
#   2. 特征工程

train = pd.read_csv('train.csv').set_index('id')
test = pd.read_csv('test.csv').set_index('id')
test['label'] = -1
all_data = pd.concat([train, test])

pipeline = Pipeline([
    ('vec', TfidfVectorizer()),
    ('clf', MultinomialNB())
    ])


X_train, X_test, y_train, y_test = train_test_split(
    train['tweet'], train['label'], test_size=0.30, random_state=28)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))
cmatrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cmatrix, ax=ax0)
sns.heatmap(cmatrix / cmatrix.sum(axis=1)[:, None], ax=ax1)
plt.show()
